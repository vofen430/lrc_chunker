# lrc_chunker

按仓库文档恢复的歌词对齐项目：输入歌曲音频和 LRC，输出词级时间戳、歌词 chunk、M0/M1 特征以及预览视频。

## 环境要求

这个项目固定要求 `conda + Python 3.8`。

```powershell
conda env create -f environment.yml
conda activate lrc-chunker-py38
pip install -e .
```

如果当前机器没有 `conda`，可先本地安装 Miniconda：

```bash
curl -fsSL -o Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$PWD/miniconda3"
"$PWD/miniconda3/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
"$PWD/miniconda3/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
"$PWD/miniconda3/bin/conda" env create -f environment.yml
"$PWD/miniconda3/bin/conda" run -n lrc-chunker-py38 pip install -e .
```

说明：

- `stable-ts 2.19.1` 官方声明 `Requires: Python >=3.8`
- `demucs` 官方声明 `Requires: Python >=3.8.0`
- `openai-whisper` 官方声明 `Requires: Python >=3.8`
- `librosa 0.10.2.post1` 支持 Python 3.8
- `matplotlib` 最新版已经要求 Python 3.10，因此这里固定到 `3.7.5`
- `imageio` 最新版已经要求 Python 3.9，因此这里固定到 `2.31.5`
- `imageio-ffmpeg 0.5.1` 用于 `imageio` 写出 mp4 预览视频
- `environment.yml` 使用 `conda` 安装 Python / Torch / `conda-forge::ffmpeg`，使用 `pip` 安装其余 Python 包，以避免当前 conda 源缺失 `matplotlib 3.7.5`，并绕开 `pytorch` channel 上旧版 `ffmpeg` 的动态库问题

## 项目结构

- `lyrics_chunker_baseline.py`: Stage 1 + Stage 2，LRC 预处理、参考歌词对齐、chunking
- `word_timing_refine.py`: Stage 3，词级时间戳修正
- `motion_m0_extract.py`: Stage 4，M0 文本/时序特征
- `motion_m1_demucs_benchmark.py`: Stage 5 + Stage 6，M1 音频特征、单视频、2x2 对比视频
- `lrc_external_processor.py`: 外部插件协议 CLI 包装层
- `src/lrc_chunker/`: 核心模块
- `docs/`: 恢复要求和验收标准
- `examples/`: 可同步的轻量样例输入（当前包含 `Memories - Conan Gray.lrc`）

## 标准流程

```powershell
python lyrics_chunker_baseline.py "song.wav" "lyrics.lrc" --language en
python word_timing_refine.py "artifacts/alignment/chunking_song_small.en.json" --audio-mix "song.wav"
python motion_m0_extract.py "artifacts/refinement/chunking_song_small.en_wordref_slow_attack.json"
python motion_m1_demucs_benchmark.py "artifacts/m0/features_text_timing_chunking_song_small.en_wordref_slow_attack.json" --audio "song.wav" --run-dir "artifacts/m1/song_small_en"
```

说明：

- Stage 2 默认会在 `artifacts/denoised/` 生成 `*_demucs_vocals.wav`，前提是 `stable-ts` 能使用已安装的 `demucs`
- Stage 2 的词文本现在完全来自 `LRC`，仅使用对齐模型输出词级时间戳
- Stage 3 默认使用 `slow_attack` profile，并默认启用 `LRC` 行首软锚点；`--audio-vocals` 和 Stage 5 的 `--vocals-path` 现在都支持省略，脚本会自动从前一阶段 JSON 元数据里复用 stem
- Stage 5 现在只保留带原音频的最终预览视频，不再落地无声 `raw.mp4`
- 测试/预览视频默认时长现在统一为 `60s`
- 如果你已有外部人声 stem，仍可手动传入 `--audio-vocals` 或 `--vocals-path`

## 已验证流程

本仓库已在本地 `Miniconda + conda Python 3.8` 环境实测通过以下真实样本：

- 输入：本地音频 `Memories - Conan Gray.wav`（不入库）
- 输入：仓库内 LRC `examples/Memories - Conan Gray.lrc`
- 产物：`artifacts/alignment/chunking_Memories_-_Conan_Gray_small.en.json`
- 产物：`artifacts/denoised/Memories_-_Conan_Gray_demucs_vocals.wav`
- 产物：`artifacts/refinement/chunking_Memories_-_Conan_Gray_small.en_wordref_slow_attack.json`
- 产物：`artifacts/m0/features_text_timing_chunking_Memories_-_Conan_Gray_small.en_wordref_slow_attack.json`
- 产物：`artifacts/m1/Memories_small_en/m1_demucs_parameter_preview.mp4`

## 输出目录

- `artifacts/alignment/`: `chunking_*.json`
- `artifacts/refinement/`: `*_wordref_*.json` 与修正报告
- `artifacts/m0/`: `features_text_timing_*.json` 与 M0 校验报告
- `artifacts/m1/`: mix / vocals 特征、M1 报告、预览视频
- `artifacts/preview/`: 2x2 对比视频

## 验收对齐

本项目的“完成”标准以以下文档为准：

- `docs/PROJECT_RESTORATION_DONE_CRITERIA.md`
- `docs/REPRO_FULL_PARAMETERS.md`
- `docs/VIDEO_MODULE_RESTORE_REQUIREMENTS.md`
- `docs/PY38_DEPENDENCY_MATRIX.md`

## 外部处理 CLI

为外部插件接入新增了文件协议 CLI，推荐安装后通过 `lrc-processor` 调用；未重新安装 editable 环境前，也可直接运行 `python lrc_external_processor.py ...`。

可用命令：

- `-A launch --job-dir "D:/Jobs/job_001"`: 读取 `request.json`，快速校验并启动 detached worker
- `version`: 输出版本号
- `self-test`: 检查核心运行时依赖
- `batch-folder --input-dir "D:/music" --output-dir "D:/out"`: 本地调试入口，按同 basename 匹配 `wav/mp3`

行为约束：

- 只有显式传 `-A` / `--ae` 时，才启用 AE 协议相关控制；默认启动不进入 AE 对接模式
- 协议以 [EXTERNAL_LRC_PROCESS_PROTOCOL.md](/home/dev/workspace/lrc_chunker/EXTERNAL_LRC_PROCESS_PROTOCOL.md) 为准
- worker 只跑 `LRC -> alignment -> chunking -> word refine -> chunk-LRC writer`
- 不进入 `M0/M1/video` 链路
- 新 LRC 只输出歌词 chunk 行；无时间戳行、非歌词行、译文行不写入结果

交付可执行文件时，推荐使用专门的 external CLI onefile 构建：

```bash
cd /home/dev/workspace/lrc_chunker
./tools/build_external_onefile.sh
./dist/lrc-processor-onefile version
./dist/lrc-processor-onefile self-test
```

这个 onefile 只面向外部 LRC 处理链路，不尝试把 `M0/M1/video` 全链路一起塞进同一个交付文件。

