# Windows Manual Test for `E:\test`

## Copy Target

Copy the whole `/test` directory to:

```text
E:\test
```

This test pack already includes 3 real `wav + lrc` pairs:

- `A COLD PLAY - The Kid LAROI.wav` / `.lrc`
- `Baby Chop - boylife,No Rome,keshi.wav` / `.lrc`
- `Cry Cry Cry - Coldplay.wav` / `.lrc`

## Included Jobs

### 1. Single-file smoke test

Job dir:

```text
E:\test\job_single
```

Input pair:

- `E:\test\Cry Cry Cry - Coldplay.wav`
- `E:\test\Cry Cry Cry - Coldplay.lrc`

### 2. Batch test

Job dir:

```text
E:\test\job_batch
```

Input pairs come from `batch_pairs.json` and cover all 3 songs in `E:\test`.

## Commands

### Check binary

```powershell
lrc-chunker.exe version
lrc-chunker.exe self-test
```

### Run single-file test

```powershell
lrc-chunker.exe -A launch --job-dir E:\test\job_single
```

### Run batch test

```powershell
lrc-chunker.exe -A launch --job-dir E:\test\job_batch
```

## Output Files

### Single

- `E:\test\job_single\status.json`
- `E:\test\job_single\output\result.lrc`

### Batch

- `E:\test\job_batch\status.json`
- `E:\test\job_batch\results\0001_A_COLD_PLAY_-_The_Kid_LAROI.lrc`
- `E:\test\job_batch\results\0002_Baby_Chop_-_boylife_No_Rome_keshi.lrc`
- `E:\test\job_batch\results\0003_Cry_Cry_Cry_-_Coldplay.lrc`

## Notes

- `launch` returns quickly and spawns a detached worker.
- Progress is tracked in `status.json`.
- Final state is one of: `completed / failed / cancelled`.
- The AE protocol path requires `-A`.
