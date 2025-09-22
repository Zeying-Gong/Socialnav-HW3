import os
import json
import tempfile
import shutil
import zipfile
import subprocess
import contextlib
from datetime import datetime

def evaluate(user_submission_file, phase_codename, test_annotation_file=None, **kwargs):
    print("Starting Evaluation.....")
    output = {"stdout": "", "stderr": ""}

    # Phase-specific parameters
    phase_params = {
        "dev": {
            "split": "val_split",
            "val_dir": "/home/zeyingg/competition/SocialNav/update_Falcon/Falcon/data/datasets/pointnav/social-hm3d/phase2:/app/Falcon/data/datasets/pointnav/social-hm3d/minival"
        },
        "minival": {
            "split": "val_split",
            "val_dir": "/home/zeyingg/competition/SocialNav/update_Falcon/Falcon/data/datasets/pointnav/social-hm3d/minival:/app/Falcon/data/datasets/pointnav/social-hm3d/minival"
        },
        "test_1": {
            "split": "test_split_v1",
            "val_dir": "/home/zeyingg/competition/SocialNav/update_Falcon/Falcon/data/datasets/pointnav/social-hm3d/phase1:/app/Falcon/data/datasets/pointnav/social-hm3d/minival"
        },
        "test_2": {
            "split": "test_split_v2",
            "val_dir": "/home/zeyingg/competition/SocialNav/update_Falcon/Falcon/data/datasets/pointnav/social-hm3d/phase2:/app/Falcon/data/datasets/pointnav/social-hm3d/minival"
        }
    }

    if phase_codename not in phase_params:
        output["stderr"] = f"Unknown phase: {phase_codename}"
        return output

    # === 日志路径 ===
    base_log_dir = kwargs.get("save_dir", "/mnt/nvme1/zeyingg/robosense_submissions")
    phase_dir = os.path.join(base_log_dir, phase_codename)
    submission_meta = kwargs.get("submission_metadata", {})
    team_name = submission_meta.get("participant_team_name", "unknown_team").replace(" ", "_")
    submission_id = submission_meta.get("id", "unknown_id")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = os.path.join(phase_dir, team_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"submission_{submission_id}_{timestamp}.log")

    filename = os.path.basename(user_submission_file)
    if filename.endswith(".zip"):
        submission_type = "code_zip"
        submission_dir = tempfile.mkdtemp(dir=os.path.abspath("./tmp"))
        with zipfile.ZipFile(user_submission_file, "r") as zip_ref:
            zip_ref.extractall(submission_dir)
        run_command = ["bash", "input/run.sh"]

    elif filename.endswith(".json"):
        if phase_codename == "test_2":
            output["stderr"] = "Phase 2 does not support action submission!"
            return output
        else:
            # submission_type = "replay_json"
            submission_dir = tempfile.mkdtemp(dir=os.path.abspath("./tmp"))
            shutil.copy(user_submission_file, os.path.join(submission_dir, "actions.json"))
            run_command = ["bash", "-c", "source activate falcon && cd /app/Falcon/ && python -u -m habitat_baselines.eval --config-name=social_nav_v2/falcon_hm3d_replay.yaml habitat_baselines.eval.split=minival"]

    else:
        output["stderr"] = "Submission file must be either a .zip or .json"
        return output

    BASE_IMAGE = "robosense_socialnav:v0.7"
    hm3d_dir = "/mnt/nvme2/zeyingg/versioned_data/hm3d-0.2"
    data_dir = "/home/zeyingg/competition/SocialNav/Falcon/data"
    container_name = f"eval_container_{os.getpid()}"
    docker_result_path = "/app/Falcon/output/result.json"
    host_result_dir = tempfile.mkdtemp(dir=os.path.abspath("./tmp"))
    host_result_path = os.path.join(host_result_dir, "result.json")

    try:
        with open(log_path, "w") as f_log, contextlib.redirect_stdout(f_log), contextlib.redirect_stderr(f_log):
            # 获取GPU设备号，默认为3，可通过环境变量CUDA_VISIBLE_DEVICES设置
            gpu_device = os.environ.get("CUDA_VISIBLE_DEVICES", "3")
            docker_cmd = [
                "docker", "create", "--name", container_name,
                "--gpus", f"device={gpu_device}",  # 使用环境变量指定的GPU
                "--runtime=nvidia",
                "-e", "NVIDIA_DRIVER_CAPABILITIES=all",
                "-e", "EGL_PLATFORM=surfaceless",
                "-w", "/app/Falcon",
                "-v", f"{submission_dir}:/app/Falcon/input:ro",
                "-v", f"{hm3d_dir}/hm3d:/mnt/nvme2/zeyingg/versioned_data/hm3d-0.2/hm3d:ro",
                "-v", "/home/zeyingg/competition/SocialNav/Falcon/data/hab3_bench_assets:/app/Falcon/data/hab3_bench_assets:ro",
                "-v", "/home/zeyingg/competition/SocialNav/Falcon/data/humanoids:/app/Falcon/data/humanoids:ro",
                "-v", "/home/zeyingg/competition/SocialNav/Falcon/data/robots:/app/Falcon/data/robots:ro",
                "-v", "/home/zeyingg/competition/SocialNav/Falcon/data/scene_datasets:/app/Falcon/data/scene_datasets:ro",
                "-v", "/home/zeyingg/competition/SocialNav/Falcon/data/versioned_data:/app/Falcon/data/versioned_data:ro",
                "-v", "/home/zeyingg/competition/SocialNav/Falcon/data/datasets/pointnav/social-hm3d/train:/app/Falcon/data/datasets/pointnav/social-hm3d/train:ro"
            ]

            
            if "val_dir" in phase_params[phase_codename]:
                docker_cmd += ["-v", phase_params[phase_codename]["val_dir"]]

            docker_cmd += [BASE_IMAGE] + run_command

            print("[INFO] Creating container...")
            subprocess.run(docker_cmd, check=True)

            print("[INFO] Starting container...")
            try:
                start_proc = subprocess.run(
                    ["docker", "start", "-a", container_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                print(start_proc.stdout)
                if start_proc.returncode != 0:
                    print(f"[ERROR] Docker container exited with code {start_proc.returncode}")
                    print("[STDERR]", start_proc.stderr)
                    output["stderr"] += f"\n[Docker Error]\n{start_proc.stderr}"
            except subprocess.CalledProcessError as e:
                print(f"[EXCEPTION] Failed to start Docker container: {e}")
                output["stderr"] += f"\n[Start Container Exception]\n{str(e)}"


            print("[INFO] Trying to copy result.json...")
            try:
                subprocess.run(["docker", "cp", f"{container_name}:{docker_result_path}", host_result_path], check=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Could not copy result.json: {e}")

            subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL)

            if not os.path.exists(host_result_path):
                raise FileNotFoundError("result.json not found after Docker run.")

            with open(host_result_path, "r") as f:
                result_dict = json.load(f)

            required_keys = ["SR", "SPL", "PSC", "H-Coll"]
            if not all(k in result_dict for k in required_keys):
                raise ValueError("result.json missing required metrics")

            for key in required_keys:
                if not isinstance(result_dict[key], (int, float)):
                    raise TypeError(f"Metric '{key}' must be numeric")

            SR, SPL, PSC, HColl = result_dict["SR"], result_dict["SPL"], result_dict["PSC"], result_dict["H-Coll"]
            Total = round(0.4 * SR + 0.3 * SPL + 0.3 * PSC, 4)

            output["result"] = [{
                "split": phase_params[phase_codename]["split"],
                "show_to_participant": True,
                "accuracies": {
                    "SR": round(SR, 4), "SPL": round(SPL, 4), "PSC": round(PSC, 4),
                    "H-Coll": round(HColl, 4), "Total": Total
                }
            }]

    except Exception as e:
        output["result"] = []
        output["stderr"] = str(e)
    finally:
        if "submission_dir" in locals():
            shutil.rmtree(submission_dir, ignore_errors=True)
        if "host_result_dir" in locals():
            shutil.rmtree(host_result_dir, ignore_errors=True)

        if os.path.exists(log_path):
            with open(log_path, "r") as f_log:
                full_log = f_log.read()

            # 优先返回末尾错误信息，如果太短就给完整日志
            if len(full_log) > 50000:
                output["stdout"] = full_log[-50000:]
            else:
                output["stdout"] = full_log

    return output
