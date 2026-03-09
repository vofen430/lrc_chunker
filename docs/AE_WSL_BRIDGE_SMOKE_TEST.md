# AE WSL Bridge Smoke Test

## Goal

This document defines the minimal validation procedure for the AE <-> WSL bridge.

It is not for timing evaluation and not for alignment quality evaluation.

It only answers one question:

- can the Windows side successfully trigger the WSL-side processor and observe a completed result through the file bridge?

## What Is Being Tested

The smoke test validates the whole bridge chain:

1. Windows can call `wsl.exe`
2. `wsl.exe` can run the WSL launcher
3. the launcher can validate the job and start detached work
4. WSL can write `status.json`
5. Windows can read `status.json` through:
   - `\\wsl.localhost\Ubuntu\...`
6. WSL can write `result.lrc`
7. Windows can read the final `result.lrc`

## Test Module

Use:

- [wsl_bridge_smoke.py](/home/dev/workspace/lrc_chunker/tools/wsl_bridge_smoke.py)
- [Test-WslNonBlocking.ps1](/home/dev/workspace/lrc_chunker/tools/Test-WslNonBlocking.ps1)

The module will:

- create a temporary job under `/tmp/wsl_bridge_smoke_job`
- copy bundled test input files
- write `request.json`
- call the stable launcher
- poll `status.json`
- verify that `result.lrc` exists
- print a machine-readable JSON summary

The PowerShell helper will:

- use the bundled WSL test files directly
- write a fresh single-job `request.json`
- launch the stable WSL bridge entrypoint
- verify that launch returned quickly instead of blocking
- verify that `status.json` advances in the background
- optionally wait for terminal completion
- print a machine-readable JSON summary

## Modes

### Quick mode

Default mode.

Purpose:

- validate the bridge itself
- finish quickly
- avoid full `stable-ts + demucs` runtime cost

Implementation:

- `denoiser = none`
- `alignment_backend = lrc`

Use this first.

### Full mode

Purpose:

- validate the real production path

Implementation:

- `denoiser = auto`
- `stable-ts`
- `demucs`
- normal refine path

This can take minutes.

## Windows-side Debug Commands

### 1. Quick smoke test

```powershell
wsl.exe -d Ubuntu bash -lc "python3 /home/dev/workspace/lrc_chunker/tools/wsl_bridge_smoke.py --timeout 30"
```

### 2. Read bridge status

```powershell
Get-Content "\\wsl.localhost\Ubuntu\tmp\wsl_bridge_smoke_job\status.json"
```

### 3. Read output LRC

```powershell
Get-Content "\\wsl.localhost\Ubuntu\tmp\wsl_bridge_smoke_job\output\result.lrc"
```

### 4. Read logs

```powershell
Get-Content "\\wsl.localhost\Ubuntu\tmp\wsl_bridge_smoke_job\stdout.log"
Get-Content "\\wsl.localhost\Ubuntu\tmp\wsl_bridge_smoke_job\stderr.log"
```

### 5. Full real-path smoke test

```powershell
wsl.exe -d Ubuntu bash -lc "python3 /home/dev/workspace/lrc_chunker/tools/wsl_bridge_smoke.py --full --timeout 1800"
```

### 6. One-click non-blocking test

Recommended when the goal is to validate AE-side listeners without blocking the caller:

```powershell
powershell -ExecutionPolicy Bypass -File "\\wsl.localhost\Ubuntu\home\dev\workspace\lrc_chunker\tools\Test-WslNonBlocking.ps1" -Full
```

Optional flags:

- `-ObserveSeconds 30`: watch longer before concluding
- `-WaitForCompletion`: keep polling until terminal state
- `-JobDirWsl /tmp/wsl_nonblocking_probe_job_2`: use a different temp job

## Success Criteria

Treat the bridge as working only if all of the following are true:

- the command exits normally
- JSON output contains:
  - `"ok": true`
- JSON output contains:
  - `"state": "completed"`
- `status.json` is readable from Windows
- `result.lrc` is readable from Windows

For the non-blocking PowerShell helper, also require:

- `non_blocking_launch_pass = true`
- `background_progress_pass = true`

## Example Successful Result

Typical key fields:

```json
{
  "ok": true,
  "launcher_exit_code": 0,
  "state": "completed",
  "status_path": "/tmp/wsl_bridge_smoke_job/status.json",
  "result_lrc_path": "/tmp/wsl_bridge_smoke_job/output/result.lrc"
}
```

Typical key fields for the one-click non-blocking helper:

```json
{
  "ok": true,
  "test_kind": "non_blocking_full",
  "launch_exit_code": 0,
  "launch_elapsed_ms": 412,
  "non_blocking_launch_pass": true,
  "background_progress_pass": true,
  "final_state": "running"
}
```

## Failure Interpretation

### Case 1. `launcher_exit_code != 0`

Meaning:

- WSL launcher failed before detached work was accepted

Check:

- `launcher_stdout`
- `launcher_stderr`

### Case 2. `state = queued` and never changes

Meaning:

- request was accepted
- but detached work did not continue

This used to happen with the old WSL bridge path.

Current recommended launcher:

- [ae_launch_wsl.sh](/home/dev/workspace/lrc_chunker/tools/ae_launch_wsl.sh)

must be used instead of directly calling:

- `lrc-processor -A launch`

### Case 3. `state = failed`

Meaning:

- job started
- worker reached a terminal error

Check:

- `status.error_code`
- `status.error_message`
- `stdout.log`
- `stderr.log`

### Case 4. `ok = false` but `state` is missing

Meaning:

- no valid `status.json` was produced within the timeout

Check:

- WSL launcher path
- distro name
- Python path
- permissions of the temp job directory

## Recommended Development Use

### Before AE integration starts

Run the quick smoke test once from Windows PowerShell.

### After changing bridge code

Run the quick smoke test again.

### Before shipping AE integration

Run:

1. quick smoke test
2. full smoke test

## Relationship To Other Documents

Use this document together with:

- [AE_WSL_BRIDGE_HANDOFF.md](/home/dev/workspace/lrc_chunker/docs/AE_WSL_BRIDGE_HANDOFF.md)
- [AE_WSL_BRIDGE_INTEGRATION.md](/home/dev/workspace/lrc_chunker/docs/AE_WSL_BRIDGE_INTEGRATION.md)

Those documents define the integration contract.

This document defines the bridge validation procedure.
