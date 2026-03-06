# lrc_chunker

按仓库文档恢复的歌词对齐项目：输入歌曲音频和 LRC，输出词级时间戳、歌词 chunk、M0/M1 特征以及预览视频。

## 环境要求

这个项目固定要求 `conda + Python 3.8`。

```powershell
conda env create -f environment.yml
conda activate lrc-chunker-py38
pip install -e .
```

说明：

- `stable-ts 2.19.1` 官方声明 `Requires: Python >=3.8`
- `demucs` 官方声明 `Requires: Python >=3.8.0`
- `openai-whisper` 官方声明 `Requires: Python >=3.8`
- `librosa 0.10.2.post1` 支持 Python 3.8
- `matplotlib` 最新版已经要求 Python 3.10，因此这里固定到 `3.7.5`
- `imageio` 最新版已经要求 Python 3.9，因此这里固定到 `2.31.5`

## 项目结构

- `lyrics_chunker_baseline.py`: Stage 1 + Stage 2，LRC 预处理、参考歌词对齐、chunking
- `word_timing_refine.py`: Stage 3，词级时间戳修正
- `motion_m0_extract.py`: Stage 4，M0 文本/时序特征
- `motion_m1_demucs_benchmark.py`: Stage 5 + Stage 6，M1 音频特征、单视频、2x2 对比视频
- `src/lrc_chunker/`: 核心模块
- `docs/`: 恢复要求和验收标准

## 标准流程

```powershell
python lyrics_chunker_baseline.py "song.wav" "lyrics.lrc" --model small.en --language en
python word_timing_refine.py "artifacts/alignment/chunking_song_small.en.json" --audio-mix "song.wav" --profile balanced
python motion_m0_extract.py "artifacts/refinement/chunking_song_small.en_wordref_balanced.json"
python motion_m1_demucs_benchmark.py "artifacts/m0/features_text_timing_song_small.en_wordref_balanced.json" --audio "song.wav" --run-dir "artifacts/m1/song_small_en"
```

如果已有人声 stem，可在 Stage 3 / Stage 5 额外传入 `--audio-vocals` 或 `--vocals-path`。

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

