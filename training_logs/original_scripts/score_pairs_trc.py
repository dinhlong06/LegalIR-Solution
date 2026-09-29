"""Wrapper quanh score_pairs.py: ep trust_remote_code=True khi FlagReranker nap model.

Vi sao can rieng: mot so reranker (vd Alibaba-NLP/gte-multilingual-reranker-base, kien
truc "new") doi hoi code tuy chinh tu repo HF, nhung FlagReranker (score_pairs.py dung
thang) khong tu bat trust_remote_code=True khi goi AutoModelForSequenceClassification/
AutoTokenizer. Khong sua truc tiep score_pairs.py (dang dung o nhieu job khac cung luc)
ma patch tu ben ngoai truoc khi goi main() cua no. CLI giong het score_pairs.py.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import score_pairs as _sp  # noqa: E402


def _patch_trust_remote_code():
    import FlagEmbedding.flag_reranker as fr
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    def _wrap(fn):
        def inner(*a, **kw):
            kw.setdefault("trust_remote_code", True)
            return fn(*a, **kw)
        return inner

    fr.AutoModelForSequenceClassification.from_pretrained = _wrap(
        AutoModelForSequenceClassification.from_pretrained)
    fr.AutoTokenizer.from_pretrained = _wrap(AutoTokenizer.from_pretrained)


def _build_parser():
    p = _sp.argparse.ArgumentParser()
    p.add_argument("--llm_reranker", action="store_true")
    p.add_argument("--texts", default=None)
    p.add_argument("--pool_dir", default="eval/pools/dev_baseline")
    p.add_argument("--out_dir", default="eval/scores/dev_baseline")
    p.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--channels", nargs="+",
                   default=["bm25:200", "bge:30", "e5:30", "gte:30", "bgesparse:30"])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--reserve_gb", type=float, default=0.0)
    p.add_argument("--ballast_gb", type=float, default=0.0)
    return p


if __name__ == "__main__":
    _patch_trust_remote_code()
    _sp.main(_build_parser().parse_args())
