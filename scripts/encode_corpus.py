#!/usr/bin/env python
"""Wrapper mỏng giữ đúng đường dẫn `scripts/encode_corpus.py` mà TRAINING.md mô tả.

Thực chất gọi `legalir.index.dense encode/merge` (vnembv2, pooling CLS):

    python scripts/encode_corpus.py encode --model <ckpt> --pooling cls --prefix "" \\
        --index_dir index/db_fast --out_prefix <out> --shard <i> --num_shards 4
    python scripts/encode_corpus.py merge --index_dir index/db_fast --out_prefix <out> \\
        --num_shards 4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legalir.index.dense import main  # noqa: E402

if __name__ == "__main__":
    main()
