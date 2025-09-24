# main.py
import os
import json
import time
import requests
import traceback
import portalocker
from datetime import datetime
from urllib.parse import urlparse
from eval_ai_interface import EvalAI_Interface
from evaluate import evaluate

# ======== Environment Variables ========
auth_token = os.environ["AUTH_TOKEN"]
evalai_api_server = os.environ["API_SERVER"]
queue_name = os.environ["QUEUE_NAME"]  # 保持单一队列
challenge_pk = os.environ["CHALLENGE_PK"]
save_dir = os.environ.get("SAVE_DIR", "./Socialnav-HW/Socialnav_HW_submissions")

# 本地消息路由目录 - 替代多队列的解决方案
local_queue_dir = os.path.join(save_dir, "local_queues")
minival_queue_dir = os.path.join(local_queue_dir, "minival")
general_queue_dir = os.path.join(local_queue_dir, "general")
os.makedirs(minival_queue_dir, exist_ok=True)
os.makedirs(general_queue_dir, exist_ok=True)

# Backup dirs
backup_queue_dir = os.path.join(save_dir, "backup_queue")
raw_queue_backup_dir = os.path.join(save_dir, "raw_queue_backup")
os.makedirs(backup_queue_dir, exist_ok=True)
os.makedirs(raw_queue_backup_dir, exist_ok=True)

# Download config
CHUNK_THRESHOLD = 100 * 1024 * 1024  # 100MB
CHUNK_SIZE = 1024 * 1024  # 1MB
MAX_DOWNLOAD_RETRIES = 3

gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
worker_pid = os.getpid()

# Define worker roles based on GPU ID
WORKER_ROLE = "BOTH"  # Default role for other GPUs

# Allow env override for worker role: GENERAL | MINIVAL_ONLY | BOTH
WORKER_ROLE_OVERRIDE = os.environ.get("WORKER_ROLE_OVERRIDE")
if WORKER_ROLE_OVERRIDE in {"GENERAL", "MINIVAL_ONLY", "BOTH"}:
    WORKER_ROLE = WORKER_ROLE_OVERRIDE

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][GPU:{gpu_id}][PID:{worker_pid}][ROLE:{WORKER_ROLE}] {msg}", flush=True)

def get_filename_from_url(url):
    return os.path.basename(urlparse(url).path)

def safe_write_backup(path, data):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as tmp_f:
        json.dump(data, tmp_f)
        tmp_f.flush()
        os.fsync(tmp_f.fileno())
    os.rename(tmp_path, path)

def route_message_to_local_queue(message_body, phase_codename):
    """将消息路由到本地队列文件"""
    submission_pk = message_body.get("submission_pk")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    if phase_codename == "minival":
        target_dir = minival_queue_dir
        queue_type = "minival"
    else:
        target_dir = general_queue_dir
        queue_type = "general"
    
    filename = f"{timestamp}_{submission_pk}.json"
    file_path = os.path.join(target_dir, filename)
    
    message_data = {
        "message_body": message_body,
        "phase_codename": phase_codename,
        "routed_at": datetime.now().isoformat(),
        "queue_type": queue_type
    }
    
    safe_write_backup(file_path, message_data)
    log(f"[ROUTER] Routed submission {submission_pk} ({phase_codename}) to {queue_type} queue")
    return file_path

def get_message_from_local_queue():
    """从本地队列获取消息"""
    try:
        # 仅处理自己角色允许的队列
        candidate_files = []
        if WORKER_ROLE == "MINIVAL_ONLY":
            files = [f for f in os.listdir(minival_queue_dir) if f.endswith('.json')]
            files.sort()
            candidate_files.extend([(minival_queue_dir, f) for f in files])
        elif WORKER_ROLE == "GENERAL":
            files = [f for f in os.listdir(general_queue_dir) if f.endswith('.json')]
            files.sort()
            candidate_files.extend([(general_queue_dir, f) for f in files])
        else:  # BOTH
            minival_files = [f for f in os.listdir(minival_queue_dir) if f.endswith('.json')]
            general_files = [f for f in os.listdir(general_queue_dir) if f.endswith('.json')]
            minival_files.sort()
            general_files.sort()
            # 合并并保持时间顺序（文件名带有时间戳前缀）
            for f in minival_files:
                candidate_files.append((minival_queue_dir, f))
            for f in general_files:
                candidate_files.append((general_queue_dir, f))
            candidate_files.sort(key=lambda x: x[1])

        if not candidate_files:
            return None

        # 依次尝试获取最早的任务，若文件上锁则尝试下一个
        for directory, oldest_file in candidate_files:
            file_path = os.path.join(directory, oldest_file)
            lock_path = file_path.replace('.json', '.lock')
            try:
                with portalocker.Lock(lock_path, 'w', timeout=0.1):
                    # 读取消息
                    with open(file_path, 'r') as f:
                        message_data = json.load(f)

                    # 删除文件
                    os.remove(file_path)

                    # 构造消息格式
                    message = {
                        "body": message_data["message_body"],
                        "receipt_handle": f"local_{oldest_file}"
                    }

                    log(f"[LOCAL_QUEUE] Retrieved message from local queue: {oldest_file}")
                    return message, message_data.get("phase_codename")
            except portalocker.exceptions.LockException:
                # 被其他进程锁住，尝试下一个
                continue
        # 如果都被锁，返回 None
        return None

    except Exception as e:
        log(f"[LOCAL_QUEUE] Error reading from local queue: {e}")
        return None

def download(submission, target_dir):
    url = submission["input_file"]
    filename = get_filename_from_url(url)
    submission_file_path = os.path.join(target_dir, filename)

    retries = 0
    while retries < MAX_DOWNLOAD_RETRIES:
        try:
            try:
                head = requests.head(url, allow_redirects=True, timeout=(10, 10))
                head.raise_for_status()
                file_size = int(head.headers.get("Content-Length", 0))
            except Exception:
                log("HEAD request failed, falling back to GET for file size detection.")
                file_size = 0

            if file_size > CHUNK_THRESHOLD or file_size == 0:
                with requests.get(url, stream=True, timeout=(10, 60)) as r:
                    r.raise_for_status()
                    with open(submission_file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)
            else:
                r = requests.get(url, timeout=(10, 60))
                r.raise_for_status()
                with open(submission_file_path, "wb") as f:
                    f.write(r.content)
            return submission_file_path
        except Exception as e:
            retries += 1
            log(f"Download attempt {retries}/{MAX_DOWNLOAD_RETRIES} failed: {e}")
            if retries >= MAX_DOWNLOAD_RETRIES:
                raise
            time.sleep(5 * retries)

def ensure_directory(phase_codename, team_name):
    today = datetime.now().strftime("%Y%m%d")
    team_name_safe = team_name.replace(" ", "_")
    dir_path = os.path.join(save_dir, today, phase_codename, team_name_safe)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def parse_message_body(message_body):
    if isinstance(message_body, str):
        try:
            return json.loads(message_body)
        except json.JSONDecodeError:
            log(f"Invalid message body JSON format: {message_body}")
            return {}
    return message_body or {}

def process_submission(evalai, message, phase_codename=None, is_backup=False):
    message_body = parse_message_body(message.get("body") if message else {})
    submission_pk = message_body.get("submission_pk")
    phase_pk = message_body.get("phase_pk")

    # 如果没有提供phase_codename，尝试获取
    if not phase_codename:
        try:
            if phase_pk:
                challenge_phase = evalai.get_challenge_phase_by_pk(phase_pk)
                phase_codename = challenge_phase["codename"]
        except Exception as e:
            log(f"Could not get phase_codename for phase_pk {phase_pk}: {e}")

    if not submission_pk or not phase_pk:
        log(f"Skipping invalid message: {message_body}")
        return "INVALID_MESSAGE"

    log_prefix = "[RECOVER]" if is_backup else "[PROCESS]"
    log(f"{log_prefix} Submission {submission_pk} (Phase: {phase_codename})")

    try:
        submission = evalai.get_submission_by_pk(submission_pk)
        team_name = submission.get("participant_team_name", "unknown_team")
        target_dir = ensure_directory(phase_codename or "unknown_phase", team_name)

        if submission.get("status") in ["finished", "failed", "cancelled"]:
            log(f"Submission {submission_pk} already in final status: {submission.get('status')}. Skipping.")
            return "ALREADY_FINAL_STATUS"

        if submission.get("status") == "submitted":
            evalai.update_submission_status({
                "submission": submission_pk,
                "submission_status": "RUNNING"
            })

        log(f"Downloading submission {submission_pk}")
        submission_file_path = download(submission, target_dir)
        log(f"Downloaded to {submission_file_path}")

        try:
            results = evaluate(submission_file_path, phase_codename, save_dir=save_dir, submission_meta=submission)
            evalai.update_submission_data({
                "challenge_phase": phase_pk,
                "submission": submission_pk,
                "stdout": results.get("stdout", ""),
                "stderr": results.get("stderr", ""),
                "submission_status": "FINISHED",
                "result": json.dumps(results.get("result", {})),
                "metadata": results.get("metadata", "")
            })
            log(f"Submission {submission_pk} evaluated successfully.")
            log_file = os.path.join(target_dir, f"{team_name}_{submission_pk}_evaluation_log.txt")
            with open(log_file, "w") as f_log:
                f_log.write(results.get("stdout", "") + "\n")
                if results.get("stderr", ""):
                    f_log.write("\n--- ERROR ---\n" + results.get("stderr", ""))
            return "SUCCESS"

        except Exception as e:
            log(f"Evaluation failed for submission {submission_pk}: {e}\n{traceback.format_exc()}")
            evalai.update_submission_data({
                "challenge_phase": phase_pk,
                "submission": submission_pk,
                "stdout": "",
                "stderr": str(e) + "\n" + traceback.format_exc(),
                "submission_status": "FAILED",
                "metadata": ""
            })
            return "EVALUATION_FAILED"

    except Exception as e:
        log(f"Unhandled error while processing submission {submission_pk}: {e}\n{traceback.format_exc()}")
        return "UNHANDLED_ERROR"

if __name__ == "__main__":
    log("Worker process started")

    backoff = 5
    while True:
        try:
            evalai = EvalAI_Interface(auth_token, evalai_api_server, queue_name, challenge_pk)
            log("Connected to EvalAI.")

            while True:
                # ===== 1. Backup Recovery (for tasks interrupted mid-processing) =====
                recovered_task = False
                for filename in os.listdir(backup_queue_dir):
                    if not filename.endswith(".json"):
                        continue
                    backup_path = os.path.join(backup_queue_dir, filename)
                    lock_path = backup_path.replace(".json", ".lock")

                    # 检查备份文件年龄 - 避免处理过新的文件
                    backup_age = time.time() - os.path.getmtime(backup_path)
                    if backup_age < 30:  # 30秒内的文件可能还在被其他worker处理
                        continue

                    try:
                        with portalocker.Lock(lock_path, 'w', timeout=0.1):
                            log(f"[RECOVER] Acquired lock on backup task: {filename}")
                            
                            try:
                                with open(backup_path, "r") as f:
                                    backup_data = json.load(f)
                            except json.JSONDecodeError:
                                log(f"[RECOVER] Corrupted backup file {filename}, deleting.")
                                os.remove(backup_path)
                                continue

                            # 处理备份数据
                            if "message_body" in backup_data:
                                message_body = backup_data["message_body"]
                                started_at = backup_data.get("started_at")
                                original_worker = f"GPU:{backup_data.get('worker_gpu', 'unknown')}/PID:{backup_data.get('worker_pid', 'unknown')}"
                                phase_codename = backup_data.get("phase_codename")
                                
                                log(f"[RECOVER] Task originally started at {started_at} by {original_worker}")
                                
                                # 检查任务是否运行时间过长
                                if started_at:
                                    start_time = datetime.fromisoformat(started_at)
                                    runtime = datetime.now() - start_time
                                    if runtime.total_seconds() > 8 * 3600:  # 8小时
                                        log(f"[RECOVER] WARNING: Task {message_body.get('submission_pk')} has been running for {runtime}. Consider manual intervention.")
                                
                            else:
                                # 兼容旧格式
                                message_body = backup_data
                                phase_codename = None
                                if message_body.get("phase_pk"):
                                    try:
                                        challenge_phase = evalai.get_challenge_phase_by_pk(message_body["phase_pk"])
                                        phase_codename = challenge_phase["codename"]
                                    except Exception as e:
                                        log(f"[RECOVER] Could not get phase_codename for phase_pk {message_body['phase_pk']}: {e}")

                            message = {"body": message_body, "receipt_handle": "dummy_handle"}

                            # 角色过滤
                            should_process = True
                            if WORKER_ROLE == "MINIVAL_ONLY" and phase_codename != "minival":
                                log(f"[RECOVER] Not my role (MINIVAL_ONLY), skipping submission {message_body.get('submission_pk')} (phase: {phase_codename})")
                                should_process = False
                            elif WORKER_ROLE == "GENERAL" and phase_codename == "minival":
                                log(f"[RECOVER] Not my role (GENERAL), skipping submission {message_body.get('submission_pk')} (phase: {phase_codename})")
                                should_process = False

                            if should_process:
                                # 处理恢复的任务
                                process_status = process_submission(evalai, message, phase_codename, is_backup=True)
                                log(f"[RECOVER] Processing status for {message_body.get('submission_pk')}: {process_status}")
                                
                                # 清理备份文件
                                if process_status not in ["SKIPPED", "INVALID_MESSAGE"]:
                                    os.remove(backup_path)
                                    log(f"[RECOVER] Cleaned up backup file for {filename}")
                                
                                recovered_task = True
                                break

                    except portalocker.exceptions.LockException:
                        log(f"[RECOVER] Backup task {filename} locked by another worker, skipping")
                        continue
                    except Exception as e:
                        log(f"[RECOVER] Error during backup recovery for {filename}: {e}\n{traceback.format_exc()}")
                        try:
                            if os.path.exists(backup_path):
                                os.remove(backup_path)
                        except:
                            pass
                        continue

                if recovered_task:
                    backoff = 5
                    continue

                # ===== 2. 检查本地队列 =====
                local_message_result = get_message_from_local_queue()
                if local_message_result:
                    message, phase_codename = local_message_result
                    submission_pk = parse_message_body(message.get("body")).get("submission_pk")
                    
                    backup_path = os.path.join(backup_queue_dir, f"{submission_pk}.json")
                    lock_path = backup_path.replace(".json", ".lock")

                    try:
                        with portalocker.Lock(lock_path, 'w', timeout=0.1):
                            log(f"[LOCAL_QUEUE] Processing submission {submission_pk} from local queue")
                            
                            # 备份到本地
                            backup_meta = {
                                "message_body": parse_message_body(message.get("body")),
                                "started_at": datetime.now().isoformat(),
                                "worker_gpu": gpu_id,
                                "worker_pid": worker_pid,
                                "phase_codename": phase_codename
                            }
                            safe_write_backup(backup_path, backup_meta)
                            
                            # 处理提交
                            process_status = process_submission(evalai, message, phase_codename, is_backup=False)
                            log(f"[LOCAL_QUEUE] Processing completed for {submission_pk}, status: {process_status}")
                            
                            # 清理备份文件
                            if process_status not in ["SKIPPED", "INVALID_MESSAGE"]:
                                os.remove(backup_path)
                            
                        backoff = 5
                        continue
                        
                    except portalocker.exceptions.LockException:
                        log(f"[LOCAL_QUEUE] Submission {submission_pk} is being processed by another worker")
                        continue
                    except Exception as e:
                        log(f"[LOCAL_QUEUE] Error processing submission {submission_pk}: {e}")
                        continue

                # ===== 3. SQS Polling 和消息路由 =====
                try:
                    message = evalai.get_message_from_sqs_queue()
                    if not message or not message.get("body"):
                        log(f"No new messages from EvalAI. Sleeping {backoff}s...")
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 60)
                        continue

                    message_body = parse_message_body(message.get("body"))
                    submission_pk = message_body.get("submission_pk")
                    phase_pk = message_body.get("phase_pk")
                    receipt_handle = message.get("receipt_handle")

                    # Raw queue backup
                    raw_backup_path = os.path.join(
                        raw_queue_backup_dir,
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{submission_pk or 'unknown'}.json"
                    )
                    safe_write_backup(raw_backup_path, message_body)
                    log(f"[SQS] Backed up raw SQS message to {os.path.basename(raw_backup_path)}")

                    if not submission_pk:
                        log(f"[SQS] Invalid message received (missing submission_pk): {message_body}.")
                        continue

                    # 获取phase信息并路由消息
                    phase_codename = None
                    if phase_pk:
                        try:
                            challenge_phase = evalai.get_challenge_phase_by_pk(phase_pk)
                            phase_codename = challenge_phase["codename"]
                        except Exception as e:
                            log(f"Could not get phase_codename for phase_pk {phase_pk}: {e}")

                    # 立即删除SQS消息（避免重复处理）
                    try:
                        evalai.delete_message_from_sqs_queue(receipt_handle)
                        log(f"[SQS] Deleted SQS message for submission {submission_pk}")
                    except Exception as delete_error:
                        log(f"[SQS] Warning: Failed to delete SQS message for {submission_pk}: {delete_error}")

                    # 路由到本地队列
                    route_message_to_local_queue(message_body, phase_codename)
                    
                    backoff = 5  # 重置退避时间
                    continue

                except Exception as queue_error:
                    log(f"Error polling EvalAI queue: {queue_error}\n{traceback.format_exc()}")
                    break

        except Exception as connection_error:
            log(f"Connection failed: {connection_error}\n{traceback.format_exc()}")
            time.sleep(10)