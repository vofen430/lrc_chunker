# Python 3.8 Dependency Matrix

目标：强制使用 `conda + Python 3.8` 复现本项目。

## Confirmed Core Packages

| Package | Pinned version | Why pinned here | Official source |
| --- | --- | --- | --- |
| `stable-ts` | `2.19.1` | 当前官方元数据仍支持 `Python >=3.8`，且本机已检测到该版本元数据 | https://pypi.org/project/stable-ts/ |
| `openai-whisper` | `20250625` | `stable-ts 2.19.1` 的依赖上限就是这个版本，且官方元数据支持 `Python >=3.8` | https://pypi.org/project/openai-whisper/ |
| `torch` | `2.3.1` | 需要保留 Python 3.8 轮子；这里避免追到更晚版本 | https://pypi.org/project/torch/2.3.1/ |
| `torchaudio` | `2.3.1` | 和 `torch 2.3.1` 配套 | https://pytorch.org/get-started/previous-versions/ |
| `demucs` | `4.0.1` | 官方包声明 `Python >=3.8.0`，用于可选人声分离 | https://pypi.org/project/demucs/ |
| `librosa` | `0.10.2.post1` | 官方版本支持 Python 3.8，用于 onset/beat/RMS 特征 | https://pypi.org/project/librosa/0.10.2.post1/ |
| `numpy` | `1.24.4` | Python 3.8 下稳定，避免新主线版本抬高 Python 下限 | https://pypi.org/project/numpy/1.24.4/ |
| `scipy` | `1.10.1` | 与 `librosa 0.10.2.post1` / `numpy 1.24.4` 搭配更稳 | https://pypi.org/project/scipy/1.10.1/ |
| `matplotlib` | `3.7.5` | 当前最新版已要求更高 Python；`3.7.5` 官方页面仍支持 Python 3.8 | https://pypi.org/project/matplotlib/3.7.5/ |
| `imageio` | `2.31.5` | 当前最新版已要求 Python 3.9；这里固定在较早且兼容 Python 3.8 的支线 | https://pypi.org/project/imageio/2.31.5/ |
| `Pillow` | `10.4.0` | 配合视频合成面板渲染，仍可用于 Python 3.8 | https://pypi.org/project/pillow/10.4.0/ |
| `ffmpeg` | conda package | 音频 mux 和视频拼接需要系统级二进制 | https://ffmpeg.org/ |

## Notes

- `imageio` 和 `matplotlib` 不能放开到最新版，否则会直接破坏 Python 3.8 约束。
- `stable-ts` 需要 `torch`、`torchaudio`、`openai-whisper`；这一组要一起 pin。
- `demucs` 在本项目里是可选增强，不是 baseline 主链的唯一前提。
- 由于当前工作区里没有 `conda` 可执行程序，本次没有实际创建 conda 环境；这里只完成了版本核查、环境文件和项目代码约束。
