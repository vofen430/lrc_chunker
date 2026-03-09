#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: ae_launch_wsl.sh <job_dir>" >&2
    exit 2
fi

JOB_DIR="$1"
ROOT="/home/dev/workspace/lrc_chunker"
CONDA_ROOT="/home/dev/workspace/miniconda3"
ENV_NAME="lrc-chunker-py38"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"

if [ ! -f "$CONDA_SH" ]; then
    echo "missing conda activation script: $CONDA_SH" >&2
    exit 13
fi

if [ ! -d "$JOB_DIR" ]; then
    echo "job_dir not found: $JOB_DIR" >&2
    exit 10
fi

cd "$ROOT"
source "$CONDA_SH"
conda activate "$ENV_NAME"
export PATH="$CONDA_ROOT/envs/$ENV_NAME/bin:$PATH"

python - "$JOB_DIR" "$ROOT" <<'PY'
from pathlib import Path
import os
import subprocess
import sys

from lrc_chunker.external_processor import load_job_request, _write_launch_status

job_dir = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
request = load_job_request(job_dir)
_write_launch_status(request)

stdout_log = job_dir / "stdout.log"
stderr_log = job_dir / "stderr.log"
stdout_log.parent.mkdir(parents=True, exist_ok=True)
stderr_log.parent.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
existing_pythonpath = env.get("PYTHONPATH", "")
src_dir = root / "src"
env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(src_dir)

env_root = Path("/home/dev/workspace/miniconda3/envs/lrc-chunker-py38")
processor_bin = env_root / "bin" / "lrc-processor"
if not processor_bin.is_file():
    raise SystemExit(13)

with stdout_log.open("ab") as stdout_handle, stderr_log.open("ab") as stderr_handle:
    proc = subprocess.Popen(
        [str(processor_bin), "-A", "run-worker", "--job-dir", str(job_dir)],
        stdin=subprocess.DEVNULL,
        stdout=stdout_handle,
        stderr=stderr_handle,
        cwd=str(root),
        env=env,
        close_fds=True,
        start_new_session=True,
    )
    if proc.poll() is not None:
        raise SystemExit(14)
PY
exit 0
