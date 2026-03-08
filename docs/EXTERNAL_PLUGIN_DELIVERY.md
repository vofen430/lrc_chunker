# External Plugin Delivery Guide

## Status

已完成“单文件信号体系”改造。

当前对接建议是：

- 插件只轮询 `status.json`
- `status.json` 同时提供：
  - 进度信息
  - 当前工作信息
  - 内容概述
  - 预计剩余时间
  - 完成 / 失败 / 取消等情况信号
  - 最终结果摘要
- `result.json`、`complete.flag`、`failed.flag` 现在仅作为兼容输出，可保留但插件不再需要依赖它们
- `cancel.flag` 仍保留为插件 -> worker 的取消输入信号

## 旧信号体系提供了哪些工作信息

原体系中，插件可从多个文件拿到这些信息：

### 1. `status.json`

提供：

- `state`: `queued / running / completed / failed / cancelled`
- `progress`: 0 到 100 的粗粒度进度
- `stage`: 当前阶段，如 `loading / aligning / writing_output / finalizing`
- `message`: 当前工作描述
- `heartbeat_utc`: 心跳
- `started_utc`: 开始时间
- `updated_utc`: 最近更新时间
- `result_lrc_path`: 单文件模式下的结果路径
- `error_code`
- `error_message`

### 2. `progress.jsonl`

提供：

- 历史进度事件流
- heartbeat 历史
- 状态变化历史
- 当前处理对象的阶段性快照

### 3. `result.json`

提供：

- 最终状态
- 最终输出路径
- warnings
- metrics
- batch 下每一首歌的结果项信息

### 4. `complete.flag` / `failed.flag`

提供：

- 终态信号
- 是否完成 / 失败 / 取消

## 新单文件信号如何完整保留上述信息

现在这些信息全部归并到一个文件：

- `status.json`

保留策略如下：

- 原 `status.json` 的即时状态字段全部保留
- 原 `progress.jsonl` 的“当前可视化所需信息”折叠成结构化 `detail`
- 原 `result.json` 的结果摘要折叠进 `detail.result_overview`
- 原 `complete.flag` / `failed.flag` 的终态语义折叠进：
  - `state`
  - `detail.signals`

因此插件侧只需轮询一个文件，就能同时知道：

- 现在在做什么
- 已完成多少
- 当前处理哪一项
- 上一项处理了什么
- 还要多久
- 是否结束
- 结束后结果在哪
- 结果概要是什么
- 是否失败或被取消

## CLI Entry

安装 editable 环境后，推荐使用：

```bash
lrc-processor <command> ...
```

仓库内也保留了 wrapper：

```bash
python lrc_external_processor.py <command> ...
```

已实现命令：

默认模式下，CLI 不进入 AE 对接流程。只有显式传 `-A` / `--ae` 时，才允许使用 AE 协议命令。

### `version`

```bash
lrc-processor version
```

输出机器可读版本号，例如：

```text
0.1.0
```

### `self-test`

```bash
lrc-processor self-test
```

输出依赖诊断 JSON。退出码为 `0` 表示当前环境可运行。

### `launch`

```bash
lrc-processor -A launch --job-dir "D:/Jobs/job_20260308_001"
```

行为：

- 读取 `job_dir/request.json`
- 快速校验协议版本、路径、输入文件、输出路径
- 写入初始 `status.json`
- detached 启动真实 worker
- 立刻返回，不阻塞插件 UI

### `batch-folder`

仅作为本地调试入口，不是 AE 主入口。

```bash
lrc-processor batch-folder --input-dir "D:/music" --output-dir "D:/out"
```

行为：

- 只扫描当前目录下的 `*.lrc`
- 用相同 basename 匹配音频，`wav` 优先于 `mp3`
- 输出新 LRC 到 `output-dir`
- 写 `batch_summary.json`

## Recommended Onefile Packaging

稳定交付路径使用专门的 external CLI onefile 构建，而不是把整项目所有能力一起塞进一个可执行文件。

构建命令：

```bash
cd /home/dev/workspace/lrc_chunker
./tools/build_external_onefile.sh
```

构建后验证：

```bash
./dist/lrc-processor-onefile version
./dist/lrc-processor-onefile self-test
```

说明：

- 该 onefile 面向外部插件集成场景
- 目标是 `LRC -> align -> chunk -> refine -> new LRC`
- 不包含 `M0/M1/video` 交付目标
- 不要求目标机再额外安装 Python 或补充旁路文件

## GitHub Actions Windows Build

如果本机不希望安装 Windows 打包依赖，可直接使用 GitHub Actions 的 Windows runner 构建 `.exe`。

工作流文件：

- [.github/workflows/build-windows-onefile.yml](/home/dev/workspace/lrc_chunker/.github/workflows/build-windows-onefile.yml)

行为：

- 在 `windows-latest` runner 上创建 `conda` 环境
- 安装项目依赖与 `pyinstaller`
- 运行 `pytest`
- 构建 `lrc-processor-onefile.exe`
- 运行 `version` 与 `self-test` 烟测
- 上传 artifact：`lrc-processor-onefile-windows-x86_64.exe`
- 若推送的是 `v*` tag，则自动附到 GitHub Release

Windows 构建脚本：

- [build_external_onefile.ps1](/home/dev/workspace/lrc_chunker/tools/build_external_onefile.ps1)

## Accepted JSON Formats

## 1. Single-file `request.json`

```json
{
  "protocol_version": 1,
  "job_id": "job_20260308_001",
  "created_utc": "2026-03-08T01:00:00Z",
  "input": {
    "lrc_path": "D:/music/song.lrc",
    "audio_path": "D:/music/song.mp3"
  },
  "output": {
    "result_lrc_path": "D:/jobs/job_20260308_001/result.lrc"
  },
  "options": {
    "mode": "default",
    "model": "small.en",
    "language": "en",
    "profile": "slow_attack",
    "denoiser": "auto",
    "use_lrc_anchors": true
  },
  "callback": {
    "status_file": "D:/jobs/job_20260308_001/status.json",
    "cancel_flag": "D:/jobs/job_20260308_001/cancel.flag"
  }
}
```

说明：

- 新体系下，插件只必需提供 `status_file`
- `cancel_flag` 推荐保留，用于取消任务
- `result_file` / `complete_flag` / `failed_flag` 现在是兼容字段，不再是插件必需依赖项

要求：

- 所有路径必须是绝对路径
- `protocol_version` 当前必须为 `1`
- `input.lrc_path` 和 `input.audio_path` 必须存在
- `output.result_lrc_path` 的父目录必须可写

## 2. Batch `request.json`

```json
{
  "protocol_version": 1,
  "job_id": "job_20260308_batch_001",
  "created_utc": "2026-03-08T01:00:00Z",
  "input": {
    "mode": "batch_manifest",
    "batch_manifest_path": "D:/jobs/job_20260308_batch_001/batch_pairs.json"
  },
  "output": {
    "result_dir": "D:/jobs/job_20260308_batch_001/results"
  },
  "options": {
    "mode": "default",
    "model": "small.en",
    "language": "en",
    "profile": "slow_attack",
    "denoiser": "auto",
    "use_lrc_anchors": true
  },
  "callback": {
    "status_file": "D:/jobs/job_20260308_batch_001/status.json",
    "cancel_flag": "D:/jobs/job_20260308_batch_001/cancel.flag"
  }
}
```

## 3. Accepted `batch_pairs.json`

这是插件传给 CLI 的“匹配对 JSON”标准格式。

```json
{
  "protocol_version": 1,
  "manifest_type": "ae_lrc_batch_pairs",
  "created_utc": "2026-03-08T01:00:00Z",
  "source": {
    "host": "after-effects",
    "panel": "LRC_Panel_Modular"
  },
  "batch": {
    "order_mode": "ui_order",
    "row_count": 3,
    "ready_count": 2,
    "skipped_count": 1
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
    },
    {
      "row_index": 2,
      "row_id": 102,
      "pair_state": "ready",
      "title": "Song B",
      "artist": "Artist B",
      "lrc_path": "D:/music/b.lrc",
      "audio_path": "D:/music/b.wav"
    }
  ],
  "skipped_rows": [
    {
      "row_index": 3,
      "row_id": 103,
      "pair_state": "missing_audio",
      "title": "Song C",
      "artist": "Artist C",
      "lrc_path": "D:/music/c.lrc",
      "audio_path": "",
      "reason": "audio file missing or not assigned"
    }
  ]
}
```

处理规则：

- 只处理 `items[]`
- `pair_state` 必须是 `ready`
- 顺序以 `items[]` 当前顺序为准，不重排
- `skipped_rows[]` 只保留给 UI/日志，不参与处理
- 若 `title` / `artist` 非空，CLI 会透传到中间 JSON 元数据中，不会从文件名反推覆盖它们

## Single Signal File

job 目录最小建议结构：

```text
job_20260308_001/
  request.json
  status.json
  cancel.flag
  result.lrc
  stdout.log
  stderr.log
```

batch job 最小建议结构：

```text
job_20260308_batch_001/
  request.json
  batch_pairs.json
  status.json
  cancel.flag
  stdout.log
  stderr.log
  results/
    0001_song_a.lrc
    0002_song_b.lrc
```

### `status.json`

这是插件唯一必读信号文件。

当前实现字段：

```json
{
  "protocol_version": 1,
  "job_id": "job_20260308_batch_001",
  "state": "running",
  "progress": 42,
  "stage": "aligning",
  "message": "processing 2/3: b.lrc",
  "heartbeat_utc": "2026-03-08T01:02:00Z",
  "started_utc": "2026-03-08T01:00:10Z",
  "updated_utc": "2026-03-08T01:02:00Z",
  "result_lrc_path": "",
  "error_code": "",
  "error_message": "",
  "detail": {
    "mode": "batch_manifest",
    "items_total": 3,
    "items_completed": 1,
    "current_item": {
      "row_index": 2,
      "row_id": 102,
      "title": "Song B",
      "artist": "Artist B",
      "lrc_path": "D:/music/b.lrc",
      "audio_path": "D:/music/b.wav",
      "result_lrc_path": "D:/jobs/job_20260308_batch_001/results/0002_b.lrc"
    },
    "last_completed_item": {
      "row_index": 1,
      "row_id": 101,
      "result_lrc_path": "D:/jobs/job_20260308_batch_001/results/0001_a.lrc",
      "input_line_count": 120,
      "output_line_count": 111,
      "chunk_count": 111,
      "warnings": []
    },
    "current_summary": "processing 2/3: b.lrc",
    "eta_seconds": 143,
    "eta_utc": "2026-03-08T01:04:23Z",
    "signals": {
      "is_terminal": false,
      "is_success": false,
      "is_error": false,
      "is_cancelled": false
    },
    "result_overview": {
      "result_dir": "D:/jobs/job_20260308_batch_001/results",
      "result_lrc_path": "",
      "warnings": [],
      "metrics": {},
      "items": []
    }
  }
}
```

## `status.json` 字段说明

### 顶层即时状态字段

- `state`: 当前终态或运行态
  - `queued`
  - `running`
  - `completed`
  - `failed`
  - `cancelled`
- `progress`: 粗粒度百分比，0 到 100
- `stage`: 当前阶段
  - `validating`
  - `loading`
  - `aligning`
  - `writing_output`
  - `finalizing`
- `message`: 当前工作概述
- `heartbeat_utc`: 心跳时间
- `started_utc`: 开始时间
- `updated_utc`: 最近更新时间
- `result_lrc_path`: 单文件模式下最终 LRC 路径；batch 运行中通常为空
- `error_code`
- `error_message`

### `detail` 中的结构化可视化信息

- `mode`: `single` 或 `batch_manifest`
- `items_total`: 总任务数
- `items_completed`: 已完成任务数
- `current_item`: 当前正在处理的项目
- `last_completed_item`: 最近一个已完成项目的摘要
- `current_summary`: 适合直接显示在 UI 的当前工作描述
- `eta_seconds`: 预计剩余秒数
- `eta_utc`: 预计完成时间（UTC）
- `signals`: 当前情况信号
  - `is_terminal`
  - `is_success`
  - `is_error`
  - `is_cancelled`
- `result_overview`: 最终结果摘要
  - `result_dir`
  - `result_lrc_path`
  - `warnings`
  - `metrics`
  - `items`

## AE-side Recommended Usage

插件只需要定时读取 `status.json`：

1. 读取 `state / progress / stage / message`
   - 用于进度条和状态标签
2. 读取 `detail.current_item`
   - 用于显示当前正在处理哪首歌
3. 读取 `detail.last_completed_item`
   - 用于显示最近完成项摘要
4. 读取 `detail.eta_seconds / eta_utc`
   - 用于预计剩余时间显示
5. 读取 `detail.signals`
   - 用于判断完成 / 失败 / 取消
6. 当 `state == completed` 时，读取：
   - 单文件模式：`result_lrc_path`
   - batch 模式：`detail.result_overview.result_dir` 与 `detail.result_overview.items`
7. 当 `state == failed` 或 `state == cancelled` 时，读取：
   - `error_code`
   - `error_message`

## Output Naming Rules

## 1. Single-file mode

输出文件名完全由请求中的 `output.result_lrc_path` 决定。

例如：

```text
D:/jobs/job_20260308_001/result.lrc
```

## 2. Batch mode

每个输出 LRC 文件名规则：

```text
{row_index:04d}_{safe_stem(lrc_path)}.lrc
```

示例：

```text
0001_song_a.lrc
0002_song_b.lrc
0012_Memories_-_Conan_Gray.lrc
```

说明：

- `row_index` 使用 `batch_pairs.json.items[].row_index`
- `safe_stem()` 会把非法字符转为 `_`
- 输出目录由 `output.result_dir` 决定

## Output Content Format

新 LRC 的内容格式当前固定为：

```text
[mm:ss.xxx]chunk text
```

示例：

```lrc
[00:02.318]It's been a couple months
[00:05.528]That's just about
[00:06.640]enough
[00:08.060]time
```

当前写出规则：

- 每个输出 chunk 对应一行 LRC
- 当某个原始歌词行被切成多个 chunk：
  - 第一块保留原始 LRC 行时间，作为 ground truth
  - 后续块使用该 chunk 的推断 `start`
- 当某个 chunk 跨越多原始歌词行：
  - 直接按 chunk 输出一行
  - 文本使用 `chunk.text`
- 不输出以下内容：
  - 无时间戳行
  - 非歌词行
  - 译文行
- 若某个推断时间会破坏整体单调性，当前实现会把它钳到不早于上一行时间，并在 `warnings` 中记录 `non_monotonic_chunk_timestamp:*`

## Exit Codes

`launch` 当前使用的退出码：

- `0`: 请求有效，worker 已成功启动
- `10`: `request.json` 非法
- `11`: 输入文件缺失
- `12`: 输出路径不可写
- `13`: 依赖缺失
- `14`: worker 启动失败
- `15`: 协议版本不支持

## Delivered Source of Truth

本交付文档描述的是“当前实现状态”。如需对照实现代码，主入口文件如下：

- `src/lrc_chunker/external_processor.py`
- `lrc_external_processor.py`
- `EXTERNAL_LRC_PROCESS_PROTOCOL.md`
