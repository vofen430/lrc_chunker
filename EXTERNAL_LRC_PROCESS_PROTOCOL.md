# AE LRC External Processor Protocol

> Last updated: 2026-03-07
> Status: Draft for implementation
> Scope: After Effects ExtendScript ScriptUI Panel <-> external long-running LRC processor

## 1. Goal

This document defines the integration contract between:

- AE side: `LRC_Panel_Modular.jsx` and its ExtendScript modules
- External side: a standalone process that receives an existing `(lrc + audio)` pair, performs long-running processing, and outputs a processed `.lrc`

The design priorities are:

1. Non-blocking launch from AE
2. Stable status polling from ExtendScript
3. Low implementation complexity
4. High compatibility on Windows
5. Easy debugging with plain files
6. Safe handling of 1 hour+ processing jobs

## 2. Chosen Integration Model

### 2.1 Required model

The external processor must be exposed as a **CLI program**.

The protocol uses:

- one detached CLI launch per job
- one job directory per task
- JSON files + flag files for status signaling
- AE polling via `app.scheduleTask(...)`

### 2.2 Why CLI is chosen

CLI is the preferred form because it is the best match for the current project:

- ExtendScript can launch external commands
- ExtendScript can reliably read small local files
- file-based protocols are easy to inspect manually
- no local server, port management, or IPC library is required
- failure states are easier to reconstruct after AE restart

### 2.3 Rejected models for v1

These models are explicitly not chosen for the first implementation:

- embedded HTTP server
- websocket or socket protocol
- named pipes
- long-lived daemon as the only entry point
- stdout streaming as the primary status channel

They may be added later, but the file protocol remains the source of truth.

## 3. High-Level Lifecycle

1. AE creates a unique `job_dir`
2. AE writes `request.json`
3. AE launches the external CLI in detached mode
4. External CLI returns quickly after the worker is successfully spawned
5. Worker writes `status.json` and heartbeat updates during processing
6. AE polls `status.json` and flag files at a low frequency
7. On completion, external side writes final result files and completion markers
8. AE detects `completed`, loads the new `.lrc`, and moves to the next stage

## 4. CLI Contract

### 4.1 Program name

The actual executable name is flexible. This document uses:

- `lrc-processor.exe` on Windows

If implemented in Python, Node, or another runtime, a launcher wrapper must still be provided so AE can call a single stable command.

### 4.2 Mandatory commands

The CLI must provide these commands:

#### `launch`

Starts a new job in detached mode.

Example:

```powershell
lrc-processor.exe launch --job-dir "D:\Jobs\job_20260307_153000_abcd"
```

Behavior:

- reads `request.json` from `job-dir`
- validates inputs quickly
- spawns the real worker
- returns immediately
- does not wait for processing completion

#### `version`

Example:

```powershell
lrc-processor.exe version
```

Behavior:

- prints machine-readable version text
- exits quickly

#### `self-test`

Example:

```powershell
lrc-processor.exe self-test
```

Behavior:

- checks runtime dependencies
- prints diagnostics
- exits with `0` only if the environment is healthy

### 4.3 Optional commands

These are recommended but not mandatory for v1:

#### `cancel`

```powershell
lrc-processor.exe cancel --job-dir "D:\Jobs\job_20260307_153000_abcd"
```

#### `validate-request`

```powershell
lrc-processor.exe validate-request --job-dir "D:\Jobs\job_20260307_153000_abcd"
```

### 4.4 CLI exit code rules

For `launch`:

- `0`: launch accepted, detached worker created successfully
- `10`: request invalid
- `11`: input file missing
- `12`: output path not writable
- `13`: dependency missing
- `14`: worker spawn failed
- `15`: unsupported protocol version

Important:

- `launch` exit code only reports whether the job was accepted and started
- final processing success or failure is reported only through the file protocol

## 5. Job Directory Layout

Each job must have its own folder.

Example:

```text
job_20260307_153000_abcd/
  request.json
  status.json
  status.tmp
  result.json
  result.lrc
  result.lrc.tmp
  stdout.log
  stderr.log
  complete.flag
  failed.flag
  cancel.flag
```

Rules:

- AE creates `job_dir`
- AE writes `request.json`
- external side owns all subsequent updates
- no two jobs may share a directory

## 6. File Ownership Rules

### AE-owned files

- `request.json`
- `cancel.flag` when cancellation is requested

### External-owned files

- `status.json`
- `result.json`
- `result.lrc`
- `stdout.log`
- `stderr.log`
- `complete.flag`
- `failed.flag`

## 7. Protocol Versioning

Every JSON file in this protocol must include:

```json
{
  "protocol_version": 1
}
```

Rules:

- current protocol version is `1`
- incompatible future changes must increment this version
- unknown extra fields must be ignored by both sides

## 8. `request.json` Schema

### 8.1 Required fields

```json
{
  "protocol_version": 1,
  "job_id": "20260307_153000_abcd",
  "created_utc": "2026-03-07T07:30:00Z",
  "input": {
    "lrc_path": "D:/input/song.lrc",
    "audio_path": "D:/input/song.mp3"
  },
  "output": {
    "result_lrc_path": "D:/jobs/job_20260307_153000_abcd/result.lrc"
  },
  "options": {
    "mode": "default"
  },
  "callback": {
    "status_file": "D:/jobs/job_20260307_153000_abcd/status.json",
    "result_file": "D:/jobs/job_20260307_153000_abcd/result.json",
    "complete_flag": "D:/jobs/job_20260307_153000_abcd/complete.flag",
    "failed_flag": "D:/jobs/job_20260307_153000_abcd/failed.flag",
    "cancel_flag": "D:/jobs/job_20260307_153000_abcd/cancel.flag"
  }
}
```

### 8.2 Field definitions

| Field | Type | Required | Description |
|---|---|---:|---|
| `protocol_version` | integer | yes | Protocol version. Current value is `1`. |
| `job_id` | string | yes | Unique job identifier. |
| `created_utc` | string | yes | ISO-8601 UTC timestamp. |
| `input.lrc_path` | string | yes | Source LRC file path. |
| `input.audio_path` | string | yes | Source audio file path. |
| `output.result_lrc_path` | string | yes | Final processed LRC path. Normally inside `job_dir`. |
| `options` | object | yes | Processing options. Unknown fields allowed. |
| `callback.status_file` | string | yes | Main status JSON path. |
| `callback.result_file` | string | yes | Final result JSON path. |
| `callback.complete_flag` | string | yes | Completion marker path. |
| `callback.failed_flag` | string | yes | Failure marker path. |
| `callback.cancel_flag` | string | yes | Cancellation request marker path. |

### 8.3 Path rules

- use absolute paths only
- Windows side may use `/` or `\`, but the external app should normalize internally
- paths must be UTF-8 safe
- the external app must not infer paths from the current working directory

### 8.4 Batch mode extension

The current AE project already keeps the batch table as an ordered array of rows containing:

- `lrcPath`
- `audioPath`
- `title`
- `artist`

For external CLI integration, AE should **not** pass the whole panel session JSON directly.

Instead, AE should export a dedicated manifest file:

- `batch_pairs.json`

When batch mode is used, `request.json` should use this form:

```json
{
  "protocol_version": 1,
  "job_id": "20260307_153000_abcd",
  "created_utc": "2026-03-07T07:30:00Z",
  "input": {
    "mode": "batch_manifest",
    "batch_manifest_path": "D:/jobs/job_20260307_153000_abcd/batch_pairs.json"
  },
  "output": {
    "result_dir": "D:/jobs/job_20260307_153000_abcd/results"
  },
  "options": {
    "mode": "default"
  },
  "callback": {
    "status_file": "D:/jobs/job_20260307_153000_abcd/status.json",
    "result_file": "D:/jobs/job_20260307_153000_abcd/result.json",
    "complete_flag": "D:/jobs/job_20260307_153000_abcd/complete.flag",
    "failed_flag": "D:/jobs/job_20260307_153000_abcd/failed.flag",
    "cancel_flag": "D:/jobs/job_20260307_153000_abcd/cancel.flag"
  }
}
```

In batch mode:

- `input.mode` must be `batch_manifest`
- `input.batch_manifest_path` is required
- `output.result_dir` is required
- the single-pair fields `input.lrc_path`, `input.audio_path`, and `output.result_lrc_path` should be omitted

## 8.5 `batch_pairs.json` Schema

This file is the recommended way to hand over the AE batch table to the external CLI.

It preserves:

- the user-defined row order
- row-level title and artist overrides
- ready/skipped information
- a stable machine-readable row state

### 8.5.1 Export policy

AE should export:

- all rows in original UI order
- `items` for rows that are ready to process
- `skipped_rows` for rows that exist in the table but are incomplete

Ready means:

- LRC exists
- audio exists

The current panel already models row completeness as:

- `OK`
- `Empty`
- `No LRC`
- `No Audio`

For external JSON, these UI states must be normalized to machine states:

- `OK` -> `ready`
- `Empty` -> `empty`
- `No LRC` -> `missing_lrc`
- `No Audio` -> `missing_audio`

### 8.5.2 Required structure

```json
{
  "protocol_version": 1,
  "manifest_type": "ae_lrc_batch_pairs",
  "created_utc": "2026-03-07T07:30:00Z",
  "source": {
    "host": "after-effects",
    "panel": "LRC_Panel_Modular"
  },
  "batch": {
    "order_mode": "ui_order",
    "row_count": 5,
    "ready_count": 3,
    "skipped_count": 2
  },
  "items": [
    {
      "row_index": 1,
      "row_id": 101,
      "pair_state": "ready",
      "title": "Song A",
      "artist": "Artist A",
      "lrc_path": "D:/music/a.lrc",
      "audio_path": "D:/music/a.mp3"
    }
  ],
  "skipped_rows": [
    {
      "row_index": 4,
      "row_id": 104,
      "pair_state": "missing_audio",
      "title": "Song D",
      "artist": "Artist D",
      "lrc_path": "D:/music/d.lrc",
      "audio_path": "",
      "reason": "audio file missing or not assigned"
    }
  ]
}
```

### 8.5.3 Field definitions

| Field | Type | Required | Description |
|---|---|---:|---|
| `protocol_version` | integer | yes | Protocol version. |
| `manifest_type` | string | yes | Must be `ae_lrc_batch_pairs`. |
| `created_utc` | string | yes | ISO-8601 UTC timestamp. |
| `source.host` | string | yes | Host application identifier. Use `after-effects`. |
| `source.panel` | string | yes | Panel identifier. Use `LRC_Panel_Modular`. |
| `batch.order_mode` | string | yes | Current value is `ui_order`. |
| `batch.row_count` | integer | yes | Total row count in the batch table. |
| `batch.ready_count` | integer | yes | Number of rows exported in `items`. |
| `batch.skipped_count` | integer | yes | Number of rows exported in `skipped_rows`. |
| `items` | array | yes | Ready rows only, in UI order. |
| `skipped_rows` | array | yes | Incomplete rows, in UI order. |

### 8.5.4 `items[]` row schema

| Field | Type | Required | Description |
|---|---|---:|---|
| `row_index` | integer | yes | 1-based UI order index. |
| `row_id` | integer | yes | Internal stable row id from AE session. |
| `pair_state` | string | yes | Must be `ready` for `items[]`. |
| `title` | string | yes | Row title override or empty string. |
| `artist` | string | yes | Row artist override or empty string. |
| `lrc_path` | string | yes | Absolute LRC file path. |
| `audio_path` | string | yes | Absolute audio file path. |

### 8.5.5 `skipped_rows[]` row schema

| Field | Type | Required | Description |
|---|---|---:|---|
| `row_index` | integer | yes | 1-based UI order index. |
| `row_id` | integer | yes | Internal stable row id from AE session. |
| `pair_state` | string | yes | `empty`, `missing_lrc`, or `missing_audio`. |
| `title` | string | yes | Row title override or empty string. |
| `artist` | string | yes | Row artist override or empty string. |
| `lrc_path` | string | yes | Absolute LRC path or empty string. |
| `audio_path` | string | yes | Absolute audio path or empty string. |
| `reason` | string | yes | Human-readable skip reason. |

### 8.5.6 Batch CLI expectations

When the external CLI receives `batch_pairs.json`, it must:

1. preserve `items[]` order as the authoritative processing order
2. ignore `skipped_rows[]` for actual processing
3. still validate every file path independently
4. not infer title or artist from filenames if a non-empty value is already provided

### 8.5.7 Why this format matches the current project

This schema is intentionally aligned with the current panel behavior:

- the batch table is already order-sensitive
- the panel already stores `lrcPath`, `audioPath`, `title`, and `artist`
- incomplete rows already exist in UI and need explicit treatment
- the external CLI usually only needs the ready pairs, not full panel state

## 9. `status.json` Schema

`status.json` is the primary AE polling target.

### 9.1 Required structure

```json
{
  "protocol_version": 1,
  "job_id": "20260307_153000_abcd",
  "state": "running",
  "progress": 42,
  "stage": "aligning",
  "message": "processing segment 84/200",
  "heartbeat_utc": "2026-03-07T08:12:11Z",
  "started_utc": "2026-03-07T07:31:02Z",
  "updated_utc": "2026-03-07T08:12:11Z",
  "result_lrc_path": "",
  "error_code": "",
  "error_message": ""
}
```

### 9.2 Field definitions

| Field | Type | Required | Description |
|---|---|---:|---|
| `protocol_version` | integer | yes | Protocol version. |
| `job_id` | string | yes | Must match `request.json`. |
| `state` | string | yes | See allowed values below. |
| `progress` | integer | yes | `0` to `100`. |
| `stage` | string | yes | Short current stage identifier. |
| `message` | string | yes | Human-readable short progress text. |
| `heartbeat_utc` | string | yes | Latest heartbeat time. |
| `started_utc` | string | yes | When processing actually started. |
| `updated_utc` | string | yes | When this status snapshot was written. |
| `result_lrc_path` | string | yes | Final LRC path when completed, else empty string. |
| `error_code` | string | yes | Stable machine-readable error code, else empty string. |
| `error_message` | string | yes | Human-readable failure description, else empty string. |

### 9.3 Allowed `state` values

Protocol-level states:

- `queued`
- `running`
- `completed`
- `failed`
- `cancelled`

AE UI mapping:

- `queued` -> running
- `running` -> running
- `completed` -> completed
- `failed` -> failed
- `cancelled` -> failed

If the AE panel only exposes three categories, it must still preserve the raw protocol state internally.

### 9.4 Stage naming

The exact processing pipeline may vary, but stage values should be short and stable. Recommended examples:

- `validating`
- `loading`
- `preprocessing`
- `analyzing`
- `aligning`
- `postprocessing`
- `writing_output`
- `finalizing`

## 10. `result.json` Schema

This file is written only once, at the end of the job.

```json
{
  "protocol_version": 1,
  "job_id": "20260307_153000_abcd",
  "state": "completed",
  "completed_utc": "2026-03-07T08:44:58Z",
  "result_lrc_path": "D:/jobs/job_20260307_153000_abcd/result.lrc",
  "warnings": [],
  "metrics": {
    "duration_sec": 4436,
    "input_line_count": 212,
    "output_line_count": 212
  },
  "error_code": "",
  "error_message": ""
}
```

If the job fails:

```json
{
  "protocol_version": 1,
  "job_id": "20260307_153000_abcd",
  "state": "failed",
  "completed_utc": "2026-03-07T08:44:58Z",
  "result_lrc_path": "",
  "warnings": [],
  "metrics": {},
  "error_code": "AUDIO_DECODE_FAILED",
  "error_message": "ffmpeg returned exit code 1"
}
```

## 11. Flag File Rules

Flag files are simple existence markers.

### `complete.flag`

Meaning:

- final result is ready
- `result.json` must exist
- `result.lrc` must exist
- `status.json` must already report `completed`

### `failed.flag`

Meaning:

- job ended unsuccessfully
- `result.json` must exist
- `status.json` must already report `failed` or `cancelled`

### `cancel.flag`

Meaning:

- AE requests cancellation
- external worker should check this periodically
- if honored, it must set state to `cancelled`

Flag contents are not important. A short text line is enough.

## 12. Atomic Write Requirements

This section is mandatory.

### 12.1 JSON files

When writing `status.json` or `result.json`, the external program must:

1. write to `*.tmp`
2. flush and close the file
3. rename/replace the target file atomically if possible

Example:

- write `status.tmp`
- rename to `status.json`

Reason:

- AE may poll while the external side is writing
- half-written JSON must never be visible as the final file

### 12.2 Result LRC

When writing `result.lrc`, the external program must:

1. write `result.lrc.tmp`
2. flush and close
3. rename to `result.lrc`
4. only then write `complete.flag`

## 13. Heartbeat Rules

For long-running jobs, heartbeat is mandatory.

Rules:

- update `heartbeat_utc` at least once every 30 seconds
- also update `updated_utc` whenever `status.json` changes
- heartbeat updates may happen even if `progress` does not change

Recommended:

- write heartbeat every 10 to 15 seconds

AE-side timeout guidance:

- if no `status.json` exists within 30 seconds after successful launch, mark as suspicious
- if `heartbeat_utc` is stale for more than 10 minutes, mark as stalled
- stalled jobs should not be auto-marked failed immediately; AE should surface a warning first

## 14. AE Polling Rules

Recommended polling schedule:

- first 10 seconds: every 1000 ms
- after 10 seconds: every 2000 ms
- after 5 minutes: every 5000 ms

Each poll should do only lightweight work:

1. check whether `status.json` exists
2. read and parse a small JSON file
3. update UI text or progress
4. check `complete.flag` or `failed.flag`
5. if completed, trigger the next AE stage

AE must not:

- read large logs every tick
- re-scan large directories every tick
- block on the worker process

## 15. AE State Machine Recommendation

Recommended internal AE states:

- `idle`
- `launching`
- `waiting_for_first_status`
- `running`
- `completed`
- `failed`
- `stalled`
- `post_processing`

Transitions:

1. `idle` -> `launching`
2. `launching` -> `waiting_for_first_status`
3. `waiting_for_first_status` -> `running`
4. `running` -> `completed`
5. `running` -> `failed`
6. `running` -> `stalled`
7. `completed` -> `post_processing`
8. `post_processing` -> `idle`

`post_processing` includes:

- loading `result.lrc`
- validating expected output
- applying the next workflow step in AE

## 16. Launch Semantics

This section defines the behavior AE expects from the external command.

### 16.1 Non-blocking guarantee

The `launch` command must return quickly. Target:

- normal return within 3 seconds
- hard upper limit within 10 seconds

If the processor needs a heavy runtime bootstrap, use a thin launcher process that exits after the worker has been detached successfully.

### 16.2 What "successful launch" means

`launch` may return `0` only if:

- `request.json` was parsed successfully
- required inputs exist
- the worker process was created successfully
- the worker is expected to take over status updates

It must not return `0` if it only queued a shell command that may or may not run later without verification.

## 17. Logging Rules

The external side must write:

- `stdout.log`
- `stderr.log`

Rules:

- append mode is allowed
- logs are for manual debugging, not for polling
- log lines should include timestamps when practical

AE should only expose log viewing on demand, not poll full logs continuously.

## 18. Error Code Conventions

Recommended stable error codes:

- `REQUEST_INVALID`
- `INPUT_LRC_MISSING`
- `INPUT_AUDIO_MISSING`
- `OUTPUT_PATH_INVALID`
- `DEPENDENCY_MISSING`
- `AUDIO_DECODE_FAILED`
- `LRC_PARSE_FAILED`
- `ALIGNMENT_FAILED`
- `WRITE_RESULT_FAILED`
- `CANCELLED_BY_USER`
- `UNKNOWN_INTERNAL_ERROR`

Rules:

- `error_code` must be stable and English
- `error_message` may be detailed and user-facing

## 19. Recovery Rules

### 19.1 AE restarted during processing

On panel startup, AE may scan known unfinished job directories and recover them.

Recovery logic:

- if `complete.flag` exists, treat as completed
- else if `failed.flag` exists, treat as failed
- else if `status.json` exists and heartbeat is recent, resume polling
- else if `status.json` exists but heartbeat is stale, treat as stalled

### 19.2 Partial outputs

AE must ignore:

- `result.lrc.tmp`
- `status.tmp`

Only final files count.

## 20. Security and Safety

Rules:

- the external processor must treat all paths from `request.json` as untrusted input
- never execute arbitrary shell fragments from request fields
- never overwrite unrelated files outside declared output paths
- sanitize any user-provided string before echoing it into a shell command

## 21. Minimal External Processor Requirements

The external program must implement all of the following:

1. Read `request.json` from `--job-dir`
2. Validate input files and protocol version
3. Spawn work without blocking the caller
4. Write `status.json` atomically
5. Update heartbeat periodically
6. Write `result.lrc` atomically
7. Write `result.json` at job end
8. Write `complete.flag` or `failed.flag`
9. Exit with defined launch exit codes

## 22. Minimal AE Integration Requirements

The AE side must implement all of the following:

1. Create a unique `job_dir`
2. Write `request.json`
3. Launch the CLI without blocking the UI for job runtime
4. Poll only lightweight files
5. Recognize `running`, `completed`, `failed`
6. Detect stalled jobs by heartbeat age
7. On `completed`, load `result.lrc` automatically into the next workflow stage
8. Preserve logs and job files for debugging

## 23. Example Job

### 23.1 Example `request.json`

```json
{
  "protocol_version": 1,
  "job_id": "20260307_153000_abcd",
  "created_utc": "2026-03-07T07:30:00Z",
  "input": {
    "lrc_path": "D:/music/in/song.lrc",
    "audio_path": "D:/music/in/song.flac"
  },
  "output": {
    "result_lrc_path": "D:/jobs/job_20260307_153000_abcd/result.lrc"
  },
  "options": {
    "mode": "default",
    "language": "auto",
    "preserve_tags": true
  },
  "callback": {
    "status_file": "D:/jobs/job_20260307_153000_abcd/status.json",
    "result_file": "D:/jobs/job_20260307_153000_abcd/result.json",
    "complete_flag": "D:/jobs/job_20260307_153000_abcd/complete.flag",
    "failed_flag": "D:/jobs/job_20260307_153000_abcd/failed.flag",
    "cancel_flag": "D:/jobs/job_20260307_153000_abcd/cancel.flag"
  }
}
```

### 23.2 Example intermediate `status.json`

```json
{
  "protocol_version": 1,
  "job_id": "20260307_153000_abcd",
  "state": "running",
  "progress": 61,
  "stage": "aligning",
  "message": "matching line timings",
  "heartbeat_utc": "2026-03-07T08:00:12Z",
  "started_utc": "2026-03-07T07:31:02Z",
  "updated_utc": "2026-03-07T08:00:12Z",
  "result_lrc_path": "",
  "error_code": "",
  "error_message": ""
}
```

### 23.3 Example final `result.json`

```json
{
  "protocol_version": 1,
  "job_id": "20260307_153000_abcd",
  "state": "completed",
  "completed_utc": "2026-03-07T08:44:58Z",
  "result_lrc_path": "D:/jobs/job_20260307_153000_abcd/result.lrc",
  "warnings": [],
  "metrics": {
    "duration_sec": 4436,
    "input_line_count": 212,
    "output_line_count": 214
  },
  "error_code": "",
  "error_message": ""
}
```

## 24. Normative Summary

This document makes the following final decisions:

1. The external program interface is **CLI**, not HTTP, not socket.
2. AE and the external process communicate through a **job directory file protocol**.
3. `status.json` is the primary status channel.
4. `complete.flag` and `failed.flag` are required secondary markers.
5. Atomic writes are mandatory for status and result files.
6. Heartbeat is mandatory for long-running jobs.
7. AE polls lightly and never waits synchronously for job completion.

This is the required v1 contract unless superseded by a newer protocol version.
