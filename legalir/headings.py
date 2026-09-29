"""Nạp đề mục v3-strict cho reranker Qwen3 ở tầng cuối.

File đầu vào `index/headings_v3_strict.jsonl`: mỗi dòng {"doc_id", "summary"} — chuỗi các
đề mục cấu trúc nối bằng " . ", do `scripts/build_heading_corpus_v3.py` dựng từ corpus.
Dòng summary rỗng bị bỏ. Phần DỰNG lại nằm ở scripts/ (không nằm trên đường suy luận).
"""
import json


def load_headings(path):
    """{doc_id: summary} — chỉ giữ văn bản có đề mục không rỗng."""
    heads = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("summary", "").strip():
            heads[r["doc_id"]] = r["summary"]
    return heads
