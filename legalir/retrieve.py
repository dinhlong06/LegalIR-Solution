"""Bước 1/4 — truy hồi 4 kênh: câu hỏi -> pool chunk ứng viên.

    bm25        BM25Okapi dạng ma trận            top-k (50 hoặc 300)
    bgesparse   lexical weights của bge-m3 GỐC    top-k (30 hoặc 300)
    bge         bi-encoder Vietnamese_Embedding_v2 đã FT, CLS, q_len 64
    e5          bi-encoder vietlegal-harrier-0.6b đã FT, last-token, KHÔNG tiền tố

Pool của một câu = HỢP của 4 danh sách (bỏ trùng chunk). Không RRF, không trọng số: tầng
sau (cross-encoder) chấm lại toàn bộ pool, nên việc của tầng này chỉ là ĐỪNG BỎ SÓT.

Đầu ra (một dòng JSON mỗi câu hỏi mỗi kênh):
    chan_{bm25,bge,e5,bgesparse}.jsonl.gz   {"qid","question","rows":[[chunk_id,doc_id,score]]}
    texts_fast.json.gz                       text của mọi chunk trong pool

    python -m legalir.retrieve --questions private-official.json --out_dir out/pool \
        --index_dir index/db_fast --bge_model models/bge-m3 \
        --dense_model models/vnembv2_ft_ep2 --dense_prefix index/dense/vnembv2ft_dense \
        --e5_model models/harrier_ft_ep2 --e5_prefix index/dense/harrierft_dense \
        --top_k_bm25 300 --top_k_dense 300

Chế độ ĐÀO CẶP huấn luyện CE fam (pool `train_fast`, 13/08 — chỉ dùng trong scripts/train_fam.sh,
KHÔNG nằm trên đường suy luận; mặc định hai cờ dưới giữ nguyên hành vi suy luận):
    --bge_channel bgem3   kênh bge = vector dense của bge-m3 GỐC (lấy từ chính lượt encode sparse),
                          ma trận corpus = bge-m3 gốc (CLS)
    --e5_mode e5          kênh e5 = bi-encoder họ e5 (e5_ft): SentenceTransformer mean pooling,
                          tiền tố "query: " — đúng `retrieval/encoder.py::E5Encoder` đã dựng train_fast
"""
import argparse
import gzip
import json
import os
import time

import numpy as np
import scipy.sparse as sp

from legalir.data import load_json, load_questions
from legalir.encoders import (encode_queries_bgem3, encode_queries_cls, encode_queries_e5,
                              encode_queries_st)
from legalir.index.bm25 import BM25Index
from legalir.index.dense import DenseIndex
from legalir.index.universe import check_universe

CHANNELS = ["bm25", "bge", "e5", "bgesparse"]


def encode_queries_st(model_path, questions, batch_size=64):
    """Kênh `e5`: SentenceTransformer, KHÔNG tiền tố (model nền Qwen3, không phải họ e5).

    Guard pooling (result.md §65): checkpoint lưu bằng save_pretrained mà thiếu
    modules.json/1_Pooling thì SentenceTransformer âm thầm dùng MEAN cho câu hỏi trong khi
    corpus được encode last-token — lệch im lặng, đã làm hỏng một bài nộp."""
    import torch
    from sentence_transformers import SentenceTransformer
    if os.path.isdir(model_path) and not os.path.exists(os.path.join(model_path, "modules.json")):
        raise SystemExit(f"⛔ {model_path}: thiếu modules.json/1_Pooling — pooling câu hỏi sẽ "
                         f"lệch với corpus. Chép modules.json, 1_Pooling/, "
                         f"config_sentence_transformers.json, sentence_bert_config.json.")
    model = SentenceTransformer(model_path,
                                device="cuda" if torch.cuda.is_available() else "cpu")
    out = []
    for i in range(0, len(questions), batch_size):
        out.extend(model.encode(questions[i:i + batch_size], normalize_embeddings=True))
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--index_dir", default="index/db_fast")
    ap.add_argument("--bge_model", default="models/bge-m3")
    ap.add_argument("--dense_model", required=True)
    ap.add_argument("--dense_prefix", required=True)
    ap.add_argument("--e5_model", required=True)
    ap.add_argument("--e5_prefix", required=True)
    ap.add_argument("--top_k_bm25", type=int, default=50)
    ap.add_argument("--top_k_dense", type=int, default=30)
    ap.add_argument("--q_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--bge_channel", choices=["cls", "bgem3"], default="cls",
                    help="cls = --dense_model FT (suy luận) · bgem3 = dense của bge-m3 gốc (đào cặp fam)")
    ap.add_argument("--sparse_npz", default=None,
                    help="ma trận bge-m3 sparse của corpus (mặc định <index_dir>/bge_sparse.npz)")
    ap.add_argument("--e5_mode", choices=["st", "e5"], default="st",
                    help="st = harrier last-token (suy luận) · e5 = e5_ft mean + 'query: ' (đào cặp fam)")
    a = ap.parse_args()

    import torch
    t_start = time.time()
    qids, questions = load_questions(a.questions)
    print(f"{len(qids)} câu hỏi")

    bm25 = BM25Index(a.index_dir)
    sparse_path = a.sparse_npz or os.path.join(a.index_dir, "bge_sparse.npz")
    sparse_mat = sp.load_npz(sparse_path).T.tocsr()
    sp_meta = load_json(sparse_path + ".ids.json.gz")

    t = time.time()
    m3_dense, sp_idx, sp_val = encode_queries_bgem3(a.bge_model, questions, a.batch_size)
    torch.cuda.empty_cache()
    dense_q = (m3_dense if a.bge_channel == "bgem3"
               else encode_queries_cls(a.dense_model, questions, a.q_len, a.batch_size))
    e5_q = (encode_queries_e5(a.e5_model, questions, a.batch_size) if a.e5_mode == "e5"
            else encode_queries_st(a.e5_model, questions, a.batch_size))
    t_encode = time.time() - t

    t = time.time()
    bge_index = DenseIndex(a.dense_prefix)
    bge_val, bge_idx = bge_index.search_batch(np.asarray(dense_q), a.top_k_dense)
    bge_ids, bge_aids = bge_index.chunk_ids, bge_index.aids
    del bge_index
    e5_index = DenseIndex(a.e5_prefix)
    e5_val, e5_idx = e5_index.search_batch(np.asarray(e5_q), a.top_k_dense)
    e5_ids = e5_index.chunk_ids
    del e5_index

    # Bốn kênh phải nói về CÙNG tập chunk (cổng legalir.index.universe): doc_id chỉ có ở
    # file id của kênh dense, các kênh khác tra nhờ bảng đó.
    aid_of = check_universe(bge_ids, bge_aids,
                            [("bm25", bm25.chunk_ids), ("e5", e5_ids),
                             ("bgesparse", sp_meta["chunk_ids"])])

    vocab_n = sparse_mat.shape[0]
    rows, cols, vals = [], [], []
    for j, (ix, vl) in enumerate(zip(sp_idx, sp_val)):
        for token, w in zip(ix, vl):
            token = int(token)
            if token < vocab_n:
                rows.append(j); cols.append(token); vals.append(float(w))
    qmat = sp.csr_matrix((np.asarray(vals, dtype=np.float32),
                          (np.asarray(rows), np.asarray(cols))),
                         shape=(len(questions), vocab_n))

    def row(cid, score):
        return [cid, str(aid_of[cid]), float(score)]

    chans = {c: {} for c in CHANNELS}
    pool_sizes = []
    for s in range(0, len(qids), 64):
        block = (qmat[s:s + 64] @ sparse_mat).toarray()
        for r in range(block.shape[0]):
            i = s + r
            qid = qids[i]
            sc = block[r]
            k = min(a.top_k_dense, sc.size)
            top = np.argpartition(-sc, k - 1)[:k]
            top = top[np.argsort(-sc[top])]
            chans["bgesparse"][qid] = [row(sp_meta["chunk_ids"][x], sc[x])
                                       for x in top if sc[x] > 0]
            chans["bm25"][qid] = [row(c, v)
                                  for v, c in bm25.search(questions[i], top_k=a.top_k_bm25)]
            chans["bge"][qid] = [row(bge_ids[x], v) for v, x in zip(bge_val[i], bge_idx[i])]
            chans["e5"][qid] = [row(e5_ids[x], v) for v, x in zip(e5_val[i], e5_idx[i])]
            pool_sizes.append(len({r_[0] for c in CHANNELS for r_ in chans[c][qid]}))
    t_retrieval = time.time() - t
    print(f"  encode {t_encode:.0f}s · retrieval {t_retrieval:.0f}s · "
          f"pool {sum(pool_sizes)/len(pool_sizes):.0f} chunk/câu")

    need = {r_[0] for c in CHANNELS for per in chans[c].values() for r_ in per}
    corpus_texts = load_json(os.path.join(a.index_dir, "texts.json.gz"))
    texts = {c: corpus_texts[c] for c in need if c in corpus_texts}
    if len(texts) != len(need):
        raise SystemExit(f"thiếu text cho {len(need) - len(texts)} chunk trong pool")

    os.makedirs(a.out_dir, exist_ok=True)
    for name in CHANNELS:
        path = os.path.join(a.out_dir, f"chan_{name}.jsonl.gz")
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for i, qid in enumerate(qids):
                f.write(json.dumps({"qid": qid, "question": questions[i],
                                    "rows": chans[name][qid]}, ensure_ascii=False) + "\n")
    with gzip.open(os.path.join(a.out_dir, "texts_fast.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    print(f"✅ pool: {len(need)} chunk · {time.time() - t_start:.0f}s -> {a.out_dir}")


if __name__ == "__main__":
    main()
