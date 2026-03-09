# AE WSL Bridge Handoff

This is the shortest developer handoff for integrating the AE plugin with the WSL-side `lrc-chunker` processor.

For the full design and rationale, see:

- [AE_WSL_BRIDGE_INTEGRATION.md](/home/dev/workspace/lrc_chunker/docs/AE_WSL_BRIDGE_INTEGRATION.md)
- [AE_WSL_BRIDGE_SMOKE_TEST.md](/home/dev/workspace/lrc_chunker/docs/AE_WSL_BRIDGE_SMOKE_TEST.md)

For the stable launcher, use:

- [ae_launch_wsl.sh](/home/dev/workspace/lrc_chunker/tools/ae_launch_wsl.sh)

For job templates, use:

- [ae_jobs/README.md](/home/dev/workspace/lrc_chunker/ae_jobs/README.md)
- [single request template](/home/dev/workspace/lrc_chunker/ae_jobs/templates/single/request.json.template)
- [batch request template](/home/dev/workspace/lrc_chunker/ae_jobs/templates/batch/request.json.template)
- [batch pairs template](/home/dev/workspace/lrc_chunker/ae_jobs/templates/batch/batch_pairs.json.template)

## 1. Fixed Constants

Use these constants on the AE side.

### Windows UNC root

```text
\\wsl.localhost\Ubuntu\home\dev\workspace\lrc_chunker
```

### WSL root

```text
/home/dev/workspace/lrc_chunker
```

### Jobs root

```text
/home/dev/workspace/lrc_chunker/ae_jobs
```

### Stable launch command template

```powershell
wsl.exe -d Ubuntu bash -lc "/home/dev/workspace/lrc_chunker/tools/ae_launch_wsl.sh /home/dev/workspace/lrc_chunker/ae_jobs/<job_id>"
```

The AE side should only replace `<job_id>`.

## 2. Interface Contract

### AE responsibilities

The AE plugin must:

1. generate a unique `job_id`
2. create:
   - `/home/dev/workspace/lrc_chunker/ae_jobs/<job_id>/`
3. create:
   - `input/`
   - `output/`
4. copy selected `.lrc` and audio into `input/`
5. write `request.json`
6. run the stable `wsl.exe` command
7. poll `status.json`
8. on `completed`, read the output `.lrc`
9. on user cancel, create `cancel.flag`

### Processor responsibilities

The processor bridge will:

- validate the request
- write the initial queued `status.json`
- launch detached `run-worker`
- update `status.json`
- write logs
- write output `.lrc`
- expose final success or failure in `status.json`

## 3. Output Rules

### Single mode

One input pair produces exactly one output file:

```text
<job_dir>/output/result.lrc
```

### Batch mode

Each input pair produces one independent `.lrc` file:

```text
<job_dir>/results/0001_xxx.lrc
<job_dir>/results/0002_xxx.lrc
...
```

Batch mode does not merge multiple songs into a single output LRC.

## 4. Files AE Must Read

### Progress and state

```text
<job_dir>/status.json
```

### Single output

```text
<job_dir>/output/result.lrc
```

### Batch output

Read all item outputs from:

```text
detail.result_overview.items[*].result_lrc_path
```

inside `status.json`.

## 5. `status.json` Fields AE Should Use

Read these fields:

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

Treat only `completed` as success.

## 6. Recommended Single-Job Layout

```text
/home/dev/workspace/lrc_chunker/ae_jobs/job_20260308_153045_ab12/
  input/
    song.lrc
    song.wav
  output/
    result.lrc
  request.json
  status.json
  stdout.log
  stderr.log
  cancel.flag
  work/
    ...
```

Windows view:

```text
\\wsl.localhost\Ubuntu\home\dev\workspace\lrc_chunker\ae_jobs\job_20260308_153045_ab12\
```

## 7. Recommended Development Order

### Phase 1

Implement only:

- single mode
- one input pair
- one `request.json`
- one `result.lrc`

### Phase 2

After single mode is stable, add:

- batch manifest
- multiple result `.lrc` files
- per-row UI reporting

## 8. Minimal Working Example

### Create request

Use the single template:

- [single request template](/home/dev/workspace/lrc_chunker/ae_jobs/templates/single/request.json.template)

### Launch

```powershell
wsl.exe -d Ubuntu bash -lc "/home/dev/workspace/lrc_chunker/tools/ae_launch_wsl.sh /home/dev/workspace/lrc_chunker/ae_jobs/job_20260308_153045_ab12"
```

### Poll

```text
\\wsl.localhost\Ubuntu\home\dev\workspace\lrc_chunker\ae_jobs\job_20260308_153045_ab12\status.json
```

### On success

Read:

```text
\\wsl.localhost\Ubuntu\home\dev\workspace\lrc_chunker\ae_jobs\job_20260308_153045_ab12\output\result.lrc
```

## 9. Practical Notes

- Do not make AE activate `conda`.
- Do not inline `ffmpeg` or PATH logic in AE.
- Do not pass Windows paths into `request.json`.
- Copy inputs into the WSL job directory first, then use WSL absolute paths.
- The stable launcher already handles:
  - project root
  - conda activation
  - PATH repair
  - request validation
  - initial `status.json`
  - detached `lrc-processor -A run-worker`

## 10. Deliverables for AE Team

The AE team only needs these four things to start:

1. [AE_WSL_BRIDGE_HANDOFF.md](/home/dev/workspace/lrc_chunker/docs/AE_WSL_BRIDGE_HANDOFF.md)
2. [AE_WSL_BRIDGE_INTEGRATION.md](/home/dev/workspace/lrc_chunker/docs/AE_WSL_BRIDGE_INTEGRATION.md)
3. [ae_launch_wsl.sh](/home/dev/workspace/lrc_chunker/tools/ae_launch_wsl.sh)
4. [ae_jobs templates](/home/dev/workspace/lrc_chunker/ae_jobs/README.md)

## 11. Validation

Before AE development starts, run the bridge smoke test:

- [AE_WSL_BRIDGE_SMOKE_TEST.md](/home/dev/workspace/lrc_chunker/docs/AE_WSL_BRIDGE_SMOKE_TEST.md)
