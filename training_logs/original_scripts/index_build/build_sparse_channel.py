"""Kênh sparse của bge-m3, chạy bằng ma trận CSR thay vì Qdrant.

Vì sao phải viết riêng: đo trên máy 2026-08-12, sparse prefetch trong Qdrant local
file mode mất **14–16 giây mỗi query** trên 184.368 chunk, so với 2,4s của dense.
Chạy 1.001 câu dev sẽ tốn hơn 4 tiếng chỉ cho một kênh.

Cách làm ở đây: rút toàn bộ sparse vector ra một lần thành scipy CSR (chunk × vocab),
rồi mỗi query chỉ là một phép nhân ma trận thưa. Kết quả giống hệt về mặt toán học
(dot product trên lexical weights), nhanh hơn khoảng ba bậc.

Chạy hai bước:
  --extract   rút CSR ra file (một lần, ~vài phút)
  (mặc định)  dùng CSR đã rút để sinh kênh sparse cho dev set
"""
import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract(args):
    from qdrant_client import QdrantClient
    import scipy.sparse as sp

    client = QdrantClient(path=args.qdrant_path)
    n = client.get_collection(args.collection).points_count
    print(f"rút sparse vector từ {args.collection}: {n} point")

    rows, cols, vals = [], [], []
    chunk_ids, aids = [], []
    offset, seen = None, 0
    t0 = time.time()
    pbar = tqdm(total=n, desc="scroll")
    while True:
        points, offset = client.scroll(
            args.collection, limit=args.scroll_batch, offset=offset,
            with_payload=["chunk_identifier", "aid"], with_vectors=["sparse"])
        if not points:
            break
        for p in points:
            sv = (p.vector or {}).get("sparse")
            r = len(chunk_ids)
            chunk_ids.append(p.payload["chunk_identifier"])
            aids.append(str(p.payload["aid"]))
            if sv is not None:
                idx = list(sv.indices)
                val = list(sv.values)
                rows.extend([r] * len(idx))
                cols.extend(idx)
                vals.extend(val)
        seen += len(points)
        pbar.update(len(points))
        if offset is None:
            break
    pbar.close()

    vocab = int(max(cols)) + 1 if cols else 1
    mat = sp.csr_matrix((np.asarray(vals, dtype=np.float32),
                         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
                        shape=(len(chunk_ids), vocab))
    os.makedirs(os.path.dirname(args.csr_path) or ".", exist_ok=True)
    sp.save_npz(args.csr_path, mat)
    with gzip.open(args.csr_path + ".ids.json.gz", "wt", encoding="utf-8") as f:
        json.dump({"chunk_ids": chunk_ids, "aids": aids}, f)
    print(f"✅ {mat.shape} nnz={mat.nnz} · {time.time()-t0:.0f}s · {args.csr_path}")


def search(args):
    import scipy.sparse as sp
    from retrieval.encoder import BGEM3Encoder

    mat = sp.load_npz(args.csr_path).T.tocsr()  # vocab × chunk, để nhân query × mat
    with gzip.open(args.csr_path + ".ids.json.gz", "rt", encoding="utf-8") as f:
        meta = json.load(f)
    chunk_ids, aids = meta["chunk_ids"], meta["aids"]
    vocab = mat.shape[0]
    print(f"CSR {mat.shape} nnz={mat.nnz}")

    with open(args.question_file, encoding="utf-8") as f:
        qdata = json.load(f)
    qids = sorted(qdata)
    if args.limit:
        qids = qids[:args.limit]
    questions = [qdata[q]["question"] for q in qids]

    enc = BGEM3Encoder(model_name=args.bge_model, use_fp16=True)
    t0 = time.time()
    rows, cols, vals = [], [], []
    for i in range(0, len(questions), 64):
        _, idxs, vs = enc.encode(questions[i:i + 64], type="query")
        for j, (ix, vv) in enumerate(zip(idxs, vs)):
            for a, b in zip(ix, vv):
                a = int(a)
                if a < vocab:
                    rows.append(i + j)
                    cols.append(a)
                    vals.append(float(b))
    qmat = sp.csr_matrix((np.asarray(vals, dtype=np.float32),
                          (np.asarray(rows), np.asarray(cols))),
                         shape=(len(questions), vocab))
    t_encode = time.time() - t0

    # cần text để rerank; lấy từ bm25_meta không có text nên đọc payload theo lô sau
    t1 = time.time()
    out = {}
    step = 64
    for s in tqdm(range(0, len(qids), step), desc="sparse search"):
        scores = (qmat[s:s + step] @ mat).toarray()
        for r in range(scores.shape[0]):
            sc = scores[r]
            k = min(args.top_k, sc.size)
            top = np.argpartition(-sc, k - 1)[:k]
            top = top[np.argsort(-sc[top])]
            out[qids[s + r]] = [[chunk_ids[t], aids[t], float(sc[t])] for t in top if sc[t] > 0]
    t_search = time.time() - t1
    print(f"encode {t_encode:.1f}s · search {t_search:.1f}s "
          f"({t_search/len(qids)*1000:.0f} ms/query)")

    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, "chan_bgesparse.jsonl.gz")
    with gzip.open(p, "wt", encoding="utf-8") as f:
        for i, qid in enumerate(qids):
            f.write(json.dumps({"qid": qid, "question": questions[i],
                                "rows": out[qid]}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "timing_bgesparse.json"), "w") as f:
        json.dump({"encode_s": t_encode, "search_s": t_search,
                   "n_queries": len(qids), "top_k": args.top_k}, f, indent=2)
    print(f"✅ {p}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--extract", action="store_true")
    p.add_argument("--qdrant_path", default="retrieval/db/bge_qdrant")
    p.add_argument("--collection", default="bge_corpus")
    p.add_argument("--csr_path", default="eval/cache/bge_sparse.npz")
    p.add_argument("--scroll_batch", type=int, default=4096)
    p.add_argument("--question_file", default="eval/splits/dev_questions.json")
    p.add_argument("--out_dir", default="eval/pools/dev_baseline")
    p.add_argument("--bge_model", default="BAAI/bge-m3")
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    (extract if a.extract else search)(a)
