#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYI="${PYI:-/home/dev/workspace/miniconda3/envs/lrc-chunker-py38/bin/pyinstaller}"

cd "$ROOT"
rm -rf build/lrc-processor-onefile dist/lrc-processor-onefile
"$PYI" --clean --noconfirm lrc-processor-onefile.spec

echo "Built: $ROOT/dist/lrc-processor-onefile"
