"""Gộp điểm chunk -> điểm văn bản: LSE T=2 trên logit -> top-10.

  - `logit`: điểm CE đã qua sigmoid nằm trong [0,1]; LSE phải cộng trong không gian LOGIT.
  - `doc_scores`: điểm(văn bản) = T·logsumexp(logit(điểm CE từng chunk)/T).
    `max` chỉ nghe chunk tốt nhất; LSE thưởng văn bản có NHIỀU chunk cùng liên quan —
    đúng chữ ký của nhãn cấp văn bản (public 0,9471 -> 0,9519).
  - `rank_documents`: cắt pool về độ sâu production, lấy top-10, phá hoà tất định
    theo (-điểm, doc_id) — giữ thứ tự duyệt = thứ tự file điểm.

Tách từ `rank.py` cũ — phép toán giữ nguyên từng dòng.
"""
import gzip
import json
import os

import numpy as np

from legalir.data import CE_CHANNELS, load_scores, pool_chunks

T_LSE = 2.0
TOP_X = 10


def logit(p):
    """Điểm CE đã qua sigmoid nằm trong [0,1]; LSE phải cộng trong không gian LOGIT."""
    z = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(z / (1 - z))


def doc_scores(chunk_scores, T=T_LSE):
    """{chunk_id: điểm} -> {doc_id: điểm LSE}."""
    by = {}
    for cid, sc in chunk_scores.items():
        by.setdefault(cid.split("_", 1)[0], []).append(sc)
    out = {}
    for d, vs in by.items():
        lg = logit(vs)
        m = lg.max()
        out[d] = float(T * (m / T + np.log(np.exp(lg / T - m / T).sum())))
    return out


def rank_documents(pool_dir, scores, channels=CE_CHANNELS, top_x=TOP_X):
    """{qid: [doc_id]} và {qid: {doc_id: điểm LSE}} ở độ sâu production."""
    restrict = pool_chunks(pool_dir, channels)
    top, scored = {}, {}
    for qid, per in scores.items():                 # giữ thứ tự file điểm
        allow = restrict.get(qid, set())
        ds = doc_scores({c: v for c, v in per.items() if c in allow})
        order = sorted(ds, key=lambda d: (-ds[d], d))
        top[qid] = [str(d) for d in order[:top_x]]
        scored[qid] = {str(d): ds[d] for d in order[:top_x]}
    return top, scored
