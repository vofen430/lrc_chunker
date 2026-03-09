# AE WSL Bridge Integration Guide

## Goal

This document defines the recommended integration model for the After Effects plugin when the actual LRC processing runs inside WSL instead of a Windows-native executable.

The bridge design solves three practical problems:

- avoid rebuilding and debugging a Windows binary for every iteration
- reuse the already validated WSL `conda + demucs + stable-ts` environment
- keep AE responsive by using the existing detached file-protocol worker

## Scope

This guide is specifically for:

- AE plugin on Windows
- WSL distro: `Ubuntu`
- project root inside WSL:
  - `/home/dev/workspace/lrc_chunker`
- Windows-side access path:
  - `\\\\wsl.localhost\\Ubuntu\\home\\dev\\workspace\\lrc_chunker`

If any of those values change, update both the launcher command and the plugin-side path constants.

## Recommended Model

The AE side does not call a Windows build of the processor.

Instead it:

1. creates a `job_dir` inside the WSL project tree
2. writes `request.json` into that `job_dir`
3. launches WSL using `wsl.exe`
4. calls a stable WSL launcher script
5. polls the WSL-side `status.json`
6. reads the output `.lrc` when the job reaches `completed`

The WSL-side processor remains the source of truth.

## Stable Launcher

Use this script as the only supported launch entry point:

- [ae_launch_wsl.sh](/home/dev/workspace/lrc_chunker/tools/ae_launch_wsl.sh)

The script does all of the following:

- changes to the project root
- activates the WSL `conda` environment
- appends the env `bin` directory to `PATH`
- validates `request.json`
- writes the initial queued `status.json`
- launches a detached:
  - `lrc-processor -A run-worker --job-dir <job_dir>`

This matters because the full pipeline needs the WSL runtime environment to expose:

- `python`
- `ffmpeg`
- `torch`
- `torchaudio`
- `demucs`
- `stable-ts`

Do not duplicate this activation logic in the AE code.
Do not have AE call `lrc-processor -A launch` directly in the WSL bridge path.

## Canonical Paths

### WSL internal paths

Use these for `request.json` and for the actual CLI launch:

- project root:
  - `/home/dev/workspace/lrc_chunker`
- jobs root:
  - `/home/dev/workspace/lrc_chunker/ae_jobs`

### Windows-side paths

Use these when the AE plugin reads or writes files directly:

- project root:
  - `\\\\wsl.localhost\\Ubuntu\\home\\dev\\workspace\\lrc_chunker`
- jobs root:
  - `\\\\wsl.localhost\\Ubuntu\\home\\dev\\workspace\\lrc_chunker\\ae_jobs`

The plugin should treat this Windows UNC path as the only supported bridge path.

## Job Directory Layout

Each AE task must have its own unique `job_dir`.

Recommended layout:

```text
/home/dev/workspace/lrc_chunker/ae_jobs/
  job_20260308_001/
    request.json
    status.json
    stdout.log
    stderr.log
    cancel.flag
    output/
      result.lrc
    work/
      ...
```

The matching Windows-side view is:

```text
\\wsl.localhost\Ubuntu\home\dev\workspace\lrc_chunker\ae_jobs\
  job_20260308_001\
    request.json
    status.json
    stdout.log
    stderr.log
    cancel.flag
    output\
      result.lrc
    work\
      ...
```

## Output Model

The output contract is already one-input to one-output.

### Single mode

One `(lrc + audio)` pair produces exactly one result file:

- `output/result.lrc`

### Batch mode

Each pair produces its own `.lrc` file in `result_dir`.

Examples:

- `0001_Cry_Cry_Cry_-_Coldplay.lrc`
- `0002_A_COLD_PLAY_-_The_Kid_LAROI.lrc`

Batch mode does not merge multiple songs into one `result.lrc`.

## AE-side Interface Contract

### Required responsibilities on AE side

The AE plugin must:

1. create a unique `job_dir`
2. write `request.json`
3. optionally delete stale `cancel.flag`
4. launch `wsl.exe`
5. poll `status.json`
6. on `completed`, read the result `.lrc`
7. on user cancel, create `cancel.flag`

### Not required on AE side

The AE plugin should not:

- activate conda itself
- build long shell command strings with inline environment setup
- inspect Python internals
- infer progress from stdout
- infer completion by file count

Progress and state must come from `status.json`.

## Recommended `request.json`

### Single mode

```json
{
  "protocol_version": 1,
  "job_id": "job_20260308_001",
  "created_utc": "2026-03-08T01:00:00Z",
  "input": {
    "mode": "single",
    "lrc_path": "/home/dev/workspace/lrc_chunker/ae_jobs/job_20260308_001/input/song.lrc",
    "audio_path": "/home/dev/workspace/lrc_chunker/ae_jobs/job_20260308_001/input/song.wav"
  },
  "output": {
    "result_lrc_path": "/home/dev/workspace/lrc_chunker/ae_jobs/job_20260308_001/output/result.lrc"
  },
  "options": {
    "model": "medium.en",
    "language": "en",
    "profile": "slow_attack",
    "denoiser": "auto",
    "use_lrc_anchors": true
  },
  "callback": {
    "status_file": "/home/dev/workspace/lrc_chunker/ae_jobs/job_20260308_001/status.json",
    "cancel_flag": "/home/dev/workspace/lrc_chunker/ae_jobs/job_20260308_001/cancel.flag"
  }
}
```

### Batch mode

Use the existing `batch_manifest` model:

- `request.json`
- `batch_pairs.json`

Each row must point to one input `lrc_path` and one input `audio_path`.

## Stable Windows Launch Command

This is the recommended command template for AE:

```powershell
wsl.exe -d Ubuntu bash -lc "/home/dev/workspace/lrc_chunker/tools/ae_launch_wsl.sh /home/dev/workspace/lrc_chunker/ae_jobs/job_20260308_001"
```

This command is intentionally stable.

The plugin should only substitute the final `job_dir`.

Do not inline `conda activate` or `PATH` mutations in the AE script itself.
Do not bypass the launcher and call `launch` directly from AE.

## AE-side Polling Path

If the job id is `job_20260308_001`, then poll:

```text
\\wsl.localhost\Ubuntu\home\dev\workspace\lrc_chunker\ae_jobs\job_20260308_001\status.json
```

On success, read:

```text
\\wsl.localhost\Ubuntu\home\dev\workspace\lrc_chunker\ae_jobs\job_20260308_001\output\result.lrc
```

## `status.json` Fields AE Should Use

The AE side should treat these fields as primary:

- `state`
- `progress`
- `stage`
- `message`
- `heartbeat_utc`
- `error_code`
- `error_message`
- `detail.items_total`
- `detail.items_completed`
- `detail.current_item`
- `detail.last_completed_item`
- `detail.eta_seconds`
- `detail.signals`
- `detail.result_overview`

### Terminal states

- `completed`
- `failed`
- `cancelled`

### Success condition

Treat the job as successful only when:

- `state == "completed"`

Then read:

- single mode: `result_lrc_path`
- batch mode: `detail.result_overview.items[*].result_lrc_path`

## AE-side Development Sequence

Implement the bridge in this order.

### 1. Hardcode the bridge roots

Windows root:

```text
\\wsl.localhost\Ubuntu\home\dev\workspace\lrc_chunker
```

WSL root:

```text
/home/dev/workspace/lrc_chunker
```

### 2. Generate a unique `job_id`

Example:

```text
job_20260308_153045_ab12
```

### 3. Create the job directory

Inside WSL root:

```text
/home/dev/workspace/lrc_chunker/ae_jobs/<job_id>/
```

Recommended subdirs:

- `input/`
- `output/`

### 4. Copy or write inputs into the job directory

Recommended:

- copy the selected `.lrc` into `input/`
- copy the selected `.wav` or `.mp3` into `input/`

This avoids path-translation ambiguity between Windows paths and WSL paths.

### 5. Write `request.json`

All paths inside the JSON should be WSL absolute paths, not Windows paths.

### 6. Launch processing

Run:

```powershell
wsl.exe -d Ubuntu bash -lc "/home/dev/workspace/lrc_chunker/tools/ae_launch_wsl.sh /home/dev/workspace/lrc_chunker/ae_jobs/<job_id>"
```

### 7. Poll `status.json`

Use `app.scheduleTask(...)` at a low frequency.

Recommended polling interval:

- `1000 ms` to `2000 ms`

### 8. Handle terminal states

If `completed`:

- read result `.lrc`
- import or forward it into the next AE stage

If `failed`:

- display `error_code`
- display `error_message`
- optionally expose `stderr.log`

If user cancels:

- create:
  - `cancel.flag`

## Suggested Error UI on AE Side

When a job fails, show:

- `state`
- `stage`
- `error_code`
- `error_message`
- job path

Also expose these files for debugging:

- `status.json`
- `stdout.log`
- `stderr.log`

## Known Runtime Notes

### 1. Full real alignment is not instant

The full WSL pipeline with:

- `stable-ts`
- `demucs`
- `slow_attack`
- `LRC anchors`

can take minutes per song.

This is normal.

### 2. `ffmpeg` must be visible in WSL `PATH`

This is already handled by:

- [ae_launch_wsl.sh](/home/dev/workspace/lrc_chunker/tools/ae_launch_wsl.sh)

Do not bypass that launcher unless you replicate the same environment setup.

### 3. Use single mode first

For first integration, only implement:

- one song
- one `request.json`
- one `result.lrc`

Then add batch support after the single flow is stable.

## Minimal AE Handoff

If you need the shortest possible developer handoff, give AE side exactly these constants:

- Windows bridge root:
  - `\\\\wsl.localhost\\Ubuntu\\home\\dev\\workspace\\lrc_chunker`
- WSL bridge root:
  - `/home/dev/workspace/lrc_chunker`
- jobs root:
  - `/home/dev/workspace/lrc_chunker/ae_jobs`
- stable launch command template:
  - `wsl.exe -d Ubuntu bash -lc "/home/dev/workspace/lrc_chunker/tools/ae_launch_wsl.sh <job_dir>"`
- status file:
  - `<job_dir>/status.json`
- single output:
  - `<job_dir>/output/result.lrc`

That is enough to implement the first working AE bridge.
