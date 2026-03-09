from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_LRC = ROOT / "test" / "Cry Cry Cry - Coldplay.lrc"
TEST_AUDIO = ROOT / "test" / "Cry Cry Cry - Coldplay.wav"
LAUNCHER = ROOT / "tools" / "ae_launch_wsl.sh"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the AE <-> WSL bridge.")
    parser.add_argument("--job-dir", type=str, default="/tmp/wsl_bridge_smoke_job")
    parser.add_argument("--full", action="store_true", help="Run the full stable-ts + demucs flow instead of the quick lrc backend path.")
    parser.add_argument("--timeout", type=int, default=90, help="Max seconds to wait for completion.")
    parser.add_argument("--poll-sec", type=float, default=1.0, help="Polling interval for status.json.")
    parser.add_argument("--keep-job-dir", action="store_true", help="Do not delete any previous contents in job_dir before running.")
    return parser


def write_request(job_dir: Path, *, full: bool) -> None:
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(TEST_LRC, input_dir / "song.lrc")
    shutil.copy2(TEST_AUDIO, input_dir / "song.wav")

    options = {
        "model": "medium.en",
        "language": "en",
        "profile": "slow_attack",
        "use_lrc_anchors": True,
    }
    if full:
        options["denoiser"] = "auto"
    else:
        options.update(
            {
                "denoiser": "none",
                "alignment_backend": "lrc",
                "use_lrc_anchors": False,
            }
        )

    payload = {
        "protocol_version": 1,
        "job_id": job_dir.name,
        "created_utc": "2026-03-08T00:00:00Z",
        "input": {
            "mode": "single",
            "lrc_path": str((input_dir / "song.lrc").resolve()),
            "audio_path": str((input_dir / "song.wav").resolve()),
        },
        "output": {
            "result_lrc_path": str((output_dir / "result.lrc").resolve()),
        },
        "options": options,
        "callback": {
            "status_file": str((job_dir / "status.json").resolve()),
            "cancel_flag": str((job_dir / "cancel.flag").resolve()),
        },
    }
    (job_dir / "request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def launch(job_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(LAUNCHER), str(job_dir)],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def poll_status(job_dir: Path, *, timeout: int, poll_sec: float) -> dict:
    status_path = job_dir / "status.json"
    deadline = time.time() + timeout
    last_payload: dict = {}
    while time.time() < deadline:
        if status_path.is_file():
            last_payload = read_json(status_path)
            if str(last_payload.get("state") or "") in {"completed", "failed", "cancelled"}:
                return last_payload
        time.sleep(poll_sec)
    return last_payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    job_dir = Path(args.job_dir).expanduser().resolve()

    if job_dir.exists() and not args.keep_job_dir:
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    if not TEST_LRC.is_file() or not TEST_AUDIO.is_file():
        print(json.dumps({"ok": False, "error": "missing bundled test inputs"}, ensure_ascii=False, indent=2))
        return 2
    if not LAUNCHER.is_file():
        print(json.dumps({"ok": False, "error": "missing launcher script"}, ensure_ascii=False, indent=2))
        return 2

    write_request(job_dir, full=bool(args.full))
    launch_result = launch(job_dir)
    status_payload = poll_status(job_dir, timeout=int(args.timeout), poll_sec=float(args.poll_sec))
    result_path = job_dir / "output" / "result.lrc"
    stdout_log = job_dir / "stdout.log"
    stderr_log = job_dir / "stderr.log"

    summary = {
        "ok": bool(status_payload.get("state") == "completed" and result_path.is_file()),
        "mode": "full" if args.full else "quick",
        "job_dir": str(job_dir),
        "launcher_exit_code": int(launch_result.returncode),
        "launcher_stdout": launch_result.stdout.strip(),
        "launcher_stderr": launch_result.stderr.strip(),
        "state": status_payload.get("state"),
        "status_path": str(job_dir / "status.json"),
        "result_lrc_path": str(result_path),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }
    if status_payload:
        summary["status"] = status_payload

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
