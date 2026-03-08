from __future__ import annotations

import os


# External CLI delivery is CPU-oriented. Disable optional accelerator branches
# that are not needed for the shipped workflow and make onefile builds brittle.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("PYTORCH_JIT", "0")
