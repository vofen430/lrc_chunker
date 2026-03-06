#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path



def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))



def main() -> int:
    _bootstrap_src()
    from lrc_chunker.word_refine import main as inner_main

    return int(inner_main())


if __name__ == "__main__":
    raise SystemExit(main())
