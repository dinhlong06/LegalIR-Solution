#!/usr/bin/env python
"""Hậu备 CLI — giống console script `legalir`, cho môi trường không `pip install .`.

    python scripts/run_inference.py <questions.json> [fam] [out_dir]

Chạy bằng interpreter nào cũng được (module cli chỉ stdlib + PyYAML, tự spawn PY_RETR/PY_CE
cho từng bước). Biến môi trường: PY_RETR · PY_CE · CORPUS_DIR · CUDA_VISIBLE_DEVICES.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legalir.cli import main          # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
