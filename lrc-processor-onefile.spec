# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata


def _optional_copy_metadata(dist_name):
    try:
        return copy_metadata(dist_name)
    except Exception:
        return []


datas = []
for dist_name in ("stable-ts", "openai-whisper", "imageio-ffmpeg", "demucs"):
    datas += _optional_copy_metadata(dist_name)
datas += collect_data_files("imageio_ffmpeg")

binaries = []
for pkg_name in ("torch", "torchaudio"):
    binaries += collect_dynamic_libs(pkg_name)

hiddenimports = [
    "imageio_ffmpeg",
    "stable_whisper",
    "stable_whisper.alignment",
    "stable_whisper.audio",
    "stable_whisper.default",
    "torchaudio",
    "torchaudio._extension",
    "whisper",
]

excludes = [
    "IPython",
    "jedi",
    "matplotlib",
    "pytest",
    "sklearn",
    "tests",
    "torch._inductor",
    "triton",
]

a = Analysis(
    ["lrc_external_processor.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["pyinstaller_runtime_external.py"],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="lrc-processor-onefile",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
