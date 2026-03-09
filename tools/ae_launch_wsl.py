#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path("/home/dev/workspace/lrc_chunker").resolve()
SRC_DIR = ROOT / "src"
ENV_BIN = Path("/home/dev/workspace/miniconda3/envs/lrc-chunker-py38/bin").resolve()
LRC_PROCESSOR = (ENV_BIN / "lrc-processor").resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AE WSL launcher without bash/conda shell wrapping.")
    parser.add_argument("job_dir", help="Absolute WSL job directory containing request.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(SRC_DIR))

    from lrc_chunker.external_processor import _write_launch_status, load_job_request

    job_dir = Path(args.job_dir).expanduser().resolve()
    request = load_job_request(job_dir)
    _write_launch_status(request)
    if not LRC_PROCESSOR.is_file():
        print(f"missing lrc-processor executable: {LRC_PROCESSOR}", file=sys.stderr)
        return 13

    stdout_log = job_dir / "stdout.log"
    stderr_log = job_dir / "stderr.log"
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(SRC_DIR)
    existing_path = env.get("PATH", "")
    env["PATH"] = f"{ENV_BIN}{os.pathsep}{existing_path}" if existing_path else str(ENV_BIN)

    with stdout_log.open("ab") as stdout_handle, stderr_log.open("ab") as stderr_handle:
        proc = subprocess.Popen(
            [str(LRC_PROCESSOR), "-A", "run-worker", "--job-dir", str(job_dir)],
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            cwd=str(ROOT),
            env=env,
            close_fds=True,
            start_new_session=True,
        )
    if proc.poll() is not None:
        return 14
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
