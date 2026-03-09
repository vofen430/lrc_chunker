# AE Jobs Template

This directory is the canonical WSL-side jobs root for the AE bridge.

Recommended runtime root:

```text
/home/dev/workspace/lrc_chunker/ae_jobs
```

Recommended Windows-side bridge path:

```text
\\wsl.localhost\Ubuntu\home\dev\workspace\lrc_chunker\ae_jobs
```

Template files are provided in:

- `templates/single/request.json.template`
- `templates/batch/request.json.template`
- `templates/batch/batch_pairs.json.template`

The AE side should generate a unique job directory per task, for example:

```text
/home/dev/workspace/lrc_chunker/ae_jobs/job_20260308_153045_ab12/
```

Recommended subdirectories per job:

- `input/`
- `output/`

The actual processor will also create:

- `status.json`
- `stdout.log`
- `stderr.log`
- `work/`
