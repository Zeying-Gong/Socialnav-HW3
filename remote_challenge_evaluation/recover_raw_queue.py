# recover_raw_queue.py
import os
import json
import sys
import time
import portalocker
import traceback
from datetime import datetime
from main import process_submission, parse_message_body, log, safe_write_backup
from eval_ai_interface import EvalAI_Interface

# Environment variables are loaded from the shell environment
AUTH_TOKEN = os.environ.get("AUTH_TOKEN")
API_SERVER = os.environ.get("API_SERVER")
QUEUE_NAME = os.environ.get("QUEUE_NAME")
CHALLENGE_PK = os.environ.get("CHALLENGE_PK")
SAVE_DIR = os.environ.get("SAVE_DIR", "./robosense-socialnav-dev/robosense_submissions")

RAW_QUEUE_BACKUP_DIR = os.path.join(SAVE_DIR, "raw_queue_backup")
BACKUP_QUEUE_DIR = os.path.join(SAVE_DIR, "backup_queue")

os.makedirs(RAW_QUEUE_BACKUP_DIR, exist_ok=True)
os.makedirs(BACKUP_QUEUE_DIR, exist_ok=True)


def recover_log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][RECOVERY_TOOL] {msg}", flush=True)

log = recover_log


def list_backups():
    log(f"Listing backups in {RAW_QUEUE_BACKUP_DIR}:")
    files = sorted(os.listdir(RAW_QUEUE_BACKUP_DIR))
    found_any = False
    for f in files:
        if f.endswith(".json"):
            print(f)
            found_any = True
    if not found_any:
        log("No .json backup files found.")


def replay_task(filename):
    raw_backup_full_path = os.path.join(RAW_QUEUE_BACKUP_DIR, filename)

    if not os.path.exists(raw_backup_full_path):
        log(f"Error: File not found: {filename}")
        return

    try:
        with open(raw_backup_full_path, "r") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError:
        log(f"Error: Corrupted JSON in {filename}. Cannot replay.")
        return

    msg_body = raw_data.get("message_body", {})
    submission_pk = msg_body.get("submission_pk")
    if not submission_pk:
        log(f"Error: Invalid message body (missing submission_pk) in {filename}. Cannot replay.")
        return

    if not AUTH_TOKEN or not API_SERVER or not QUEUE_NAME or not CHALLENGE_PK:
        log("Error: Environment variables not set (AUTH_TOKEN, API_SERVER, QUEUE_NAME, CHALLENGE_PK).")
        return

    try:
        evalai = EvalAI_Interface(AUTH_TOKEN, API_SERVER, QUEUE_NAME, CHALLENGE_PK)
        log(f"Connected to EvalAI for replaying {submission_pk}.")
    except Exception as e:
        log(f"Error connecting to EvalAI: {e}. Cannot replay.")
        return

    processing_backup_path = os.path.join(BACKUP_QUEUE_DIR, f"{submission_pk}.json")
    processing_lock_path = processing_backup_path.replace(".json", ".lock")

    try:
        with portalocker.Lock(processing_lock_path, 'w', timeout=1):
            log(f"Acquired processing lock for {submission_pk}.")

            safe_write_backup(processing_backup_path, msg_body)
            log(f"Created temporary processing backup for {submission_pk}.")

            message = {
                "body": json.dumps(msg_body),  # 必须是字符串
                "receipt_handle": None
            }

            process_status = process_submission(evalai, message, is_backup=False)
            log(f"Replay processing status for {submission_pk}: {process_status}")

            os.remove(processing_backup_path)
            os.remove(processing_lock_path)
            log(f"Cleaned up temporary processing files for {submission_pk}.")

    except portalocker.exceptions.LockException:
        log(f"Error: Submission {submission_pk} is currently locked by another worker. Skipping replay.")
    except Exception as e:
        log(f"Unexpected error during replay: {e}\n{traceback.format_exc()}")
        if os.path.exists(processing_backup_path):
            os.remove(processing_backup_path)
        if os.path.exists(processing_lock_path):
            os.remove(processing_lock_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python recover_raw_queue.py list | replay <filename>")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "list":
        list_backups()
    elif cmd == "replay" and len(sys.argv) == 3:
        replay_task(sys.argv[2])
    else:
        print("Invalid command or missing filename for replay.")
        sys.exit(1)
