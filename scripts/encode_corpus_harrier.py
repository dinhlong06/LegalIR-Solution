#!/usr/bin/env python
"""Wrapper mỏng giữ đúng đường dẫn `scripts/encode_corpus_harrier.py` mà TRAINING.md mô tả.

Thực chất gọi `legalir.index.dense encode/merge` (harrier, pooling last-token):

    python scripts/encode_corpus_harrier.py encode --model <ckpt> --pooling last_token \\
        --prefix "" --index_dir index/db_fast --out_prefix <out> --shard <i> --num_shards 4
    python scripts/encode_corpus_harrier.py merge --index_dir index/db_fast \\
        --out_prefix <out> --num_shards 4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legalir.index.dense import main  # noqa: E402

if __name__ == "__main__":
    main()
