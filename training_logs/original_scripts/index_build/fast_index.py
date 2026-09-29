"""Index nhanh: BM25 + dense + sparse chạy bằng ma trận, thay cho rank_bm25 và Qdrant.

Lý do tồn tại (đo 2026-08-12 trên 184.368 chunk, xem result.md §5):

    rank_bm25.get_scores   2.350 ms/câu   <- list comprehension Python qua 184K dict
    Qdrant dense           2.070 ms/câu
    Qdrant sparse         15.000 ms/câu
    -------------------------------------
    cùng phép toán, dạng ma trận:  ~10 ms/câu

Không đổi thuật toán, không đổi kết quả — chỉ đổi cách thực thi. Mỗi lớp ở đây đều
có bản kiểm chứng trùng khít với bản gốc (xem retrieval/verify_fast_index.py).
"""
import json
import os
import pickle
import sys
import types

import numpy as np
import scipy.sparse as sp


def _alias_vidrill():
    """bm25_index.pkl được pickle từ notebook Kaggle với package tên `ViDRILL`."""
    import retrieval.search as rs
    sys.modules.setdefault("ViDRILL", types.ModuleType("ViDRILL"))
    sys.modules.setdefault("ViDRILL.search", rs)


def _top_k_indices(scores, top_k: int):
    """Top-k chỉ số, TRÙNG KHÍT với `sorted(range(n), key=..., reverse=True)[:top_k]`.

    Bản cũ sort đầy đủ 184.368 phần tử bằng Python chỉ để lấy 200 cái đầu — đo được
    2.762 ms/query, là thành phần chậm nhất cả pipeline. `argpartition` là O(n).

    Phải cẩn thận với hoà điểm để kết quả không lệch một ly nào:
    `sorted(..., reverse=True)` của Python là sort ỔN ĐỊNH, nên các phần tử bằng điểm
    giữ nguyên thứ tự chỉ số TĂNG DẦN (reverse=True đảo phép so sánh chứ không đảo
    thứ tự các phần tử bằng nhau). Vì vậy ở ngưỡng cắt phải lấy các chỉ số NHỎ NHẤT
    trong nhóm bằng điểm — `np.flatnonzero` trả về chỉ số tăng dần nên chỉ cần cắt đầu.

    Định nghĩa nằm ở đây (không phải search.py) để đường inference nhanh không phải
    import search.py — file đó kéo theo qdrant_client, thứ image không cần nữa.
    """
    scores = np.asarray(scores)
    n = scores.shape[0]
    k = min(top_k, n)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k >= n:
        return np.lexsort((np.arange(n), -scores))

    thr = scores[np.argpartition(-scores, k - 1)[:k]].min()
    strictly = np.flatnonzero(scores > thr)
    ties = np.flatnonzero(scores == thr)
    sel = np.concatenate([strictly, ties[:k - strictly.shape[0]]])
    return sel[np.lexsort((sel, -scores[sel]))]


# --------------------------------------------------------------------------- BM25
def build_bm25_csr(pkl_path: str, out_dir: str):
    """Chuyển BM25Okapi đã pickle thành (ma trận trọng số CSR, vector idf, vocab).

    BM25Okapi.get_scores tính:
        score(q,d) = Σ_t∈q  idf[t] · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1−b+b·len_d/avgdl))

    Mẫu số chỉ phụ thuộc DOCUMENT, không phụ thuộc term. Nên toàn bộ phân số
    tf·(k1+1)/(tf+norm_d) tính sẵn được thành ma trận W[d,t] cố định, và điểm trở
    thành đúng một phép nhân ma trận thưa:  score = W @ v,  với v[t] = idf[t]·đếm(t trong query).

    Lấy idf/k1/b/doc_len thẳng từ object đã pickle ⇒ khớp tuyệt đối theo định nghĩa,
    không phải tokenize lại corpus.
    """
    _alias_vidrill()
    with open(pkl_path, "rb") as f:
        wrapper = pickle.load(f)
    bm = wrapper.bm25
    k1, b, avgdl = bm.k1, bm.b, bm.avgdl
    doc_len = np.asarray(bm.doc_len, dtype=np.float64)
    norm = k1 * (1.0 - b + b * doc_len / avgdl)          # theo document

    # float64 xuyên suốt: BM25Okapi.get_scores cộng dồn trên np.zeros(corpus_size)
    # tức float64. Để float32 thì sai số ~2e-5, đủ để đảo thứ tự các cặp gần bằng
    # điểm (đo được 1/120 lần lệch top-k). Đổi lại tốn gấp đôi bộ nhớ ma trận.
    vocab = {t: i for i, t in enumerate(bm.idf)}
    idf = np.zeros(len(vocab), dtype=np.float64)
    for t, i in vocab.items():
        idf[i] = bm.idf[t]

    nnz = sum(len(d) for d in bm.doc_freqs)
    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    vals = np.empty(nnz, dtype=np.float64)
    p = 0
    for d, freqs in enumerate(bm.doc_freqs):
        nd = norm[d]
        for t, tf in freqs.items():
            j = vocab.get(t)
            if j is None:            # term không có idf -> get_scores cộng 0
                continue
            rows[p] = d
            cols[p] = j
            vals[p] = tf * (k1 + 1.0) / (tf + nd)
            p += 1
    W = sp.csr_matrix((vals[:p], (rows[:p], cols[:p])),
                      shape=(bm.corpus_size, len(vocab)), dtype=np.float64)

    os.makedirs(out_dir, exist_ok=True)
    sp.save_npz(os.path.join(out_dir, "bm25_W.npz"), W)
    np.save(os.path.join(out_dir, "bm25_idf.npy"), idf)
    with open(os.path.join(out_dir, "bm25_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "bm25_ids.json"), "w", encoding="utf-8") as f:
        json.dump(wrapper.index_corpus, f, ensure_ascii=False)
    print(f"✅ BM25 CSR {W.shape} nnz={W.nnz} -> {out_dir}")
    return W, idf, vocab


class BM25Csr:
    """BM25Okapi tương đương, chạy bằng nhân ma trận thưa."""

    def __init__(self, index_dir: str, chunk_to_aid: dict = None):
        self.W = sp.load_npz(os.path.join(index_dir, "bm25_W.npz")).tocsc()
        self.idf = np.load(os.path.join(index_dir, "bm25_idf.npy"))
        with open(os.path.join(index_dir, "bm25_vocab.json"), encoding="utf-8") as f:
            self.vocab = json.load(f)
        with open(os.path.join(index_dir, "bm25_ids.json"), encoding="utf-8") as f:
            self.chunk_ids = json.load(f)
        self.chunk_to_aid = chunk_to_aid

    def get_scores(self, tokens):
        """Cộng dồn theo cột — nhanh hơn nhân với vector thưa vì query chỉ ~20 term."""
        score = np.zeros(self.W.shape[0], dtype=np.float64)
        for t in tokens:                       # term lặp lại thì cộng nhiều lần, như bản gốc
            j = self.vocab.get(t)
            if j is None:
                continue
            col = self.W.getcol(j)
            score[col.indices] += self.idf[j] * col.data
        return score

    def search(self, query: str, top_k: int = 5):
        scores = self.get_scores(query.split())
        idx = _top_k_indices(scores, top_k)
        return [(float(scores[i]), self.chunk_ids[i]) for i in idx]


# -------------------------------------------------------------------------- dense
def extract_dense(qdrant_path: str, collection: str, out_prefix: str,
                  scroll_batch: int = 8192):
    """Rút toàn bộ dense vector của một collection Qdrant ra ma trận numpy.

    Vector đã được chuẩn hoá lúc index (mọi encoder gọi normalize_embeddings=True),
    và Qdrant dùng distance COSINE ⇒ cosine = tích vô hướng ⇒ xếp hạng trùng khít.
    """
    from qdrant_client import QdrantClient
    from tqdm import tqdm

    client = QdrantClient(path=qdrant_path)
    n = client.get_collection(collection).points_count
    mat, chunk_ids, aids = None, [], []
    offset, pbar = None, tqdm(total=n, desc=f"scroll {collection}")
    while True:
        pts, offset = client.scroll(collection, limit=scroll_batch, offset=offset,
                                    with_payload=["chunk_identifier", "aid"],
                                    with_vectors=["dense"])
        if not pts:
            break
        for p in pts:
            v = np.asarray((p.vector or {}).get("dense"), dtype=np.float32)
            if mat is None:
                mat = np.zeros((n, v.shape[0]), dtype=np.float16)
            mat[len(chunk_ids)] = v
            chunk_ids.append(p.payload["chunk_identifier"])
            aids.append(str(p.payload["aid"]))
        pbar.update(len(pts))
        if offset is None:
            break
    pbar.close()

    mat = mat[:len(chunk_ids)]
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    np.save(out_prefix + ".npy", mat)
    with open(out_prefix + ".ids.json", "w", encoding="utf-8") as f:
        json.dump({"chunk_ids": chunk_ids, "aids": aids}, f, ensure_ascii=False)
    print(f"✅ dense {mat.shape} -> {out_prefix}.npy")
    return mat


class DenseMatrix:
    """Tìm top-k bằng một phép nhân ma trận, trên GPU nếu có."""

    def __init__(self, prefix: str, device: str = "cuda"):
        import torch
        self.torch = torch
        mat = np.load(prefix + ".npy")
        with open(prefix + ".ids.json", encoding="utf-8") as f:
            meta = json.load(f)
        self.chunk_ids = meta["chunk_ids"]
        self.aids = meta["aids"]
        self.device = device if torch.cuda.is_available() else "cpu"
        self.mat = torch.from_numpy(mat).to(self.device)
        if self.device == "cpu":
            self.mat = self.mat.float()

    def search_batch(self, qvecs: np.ndarray, top_k: int, block: int = 512):
        """Chia khối theo QUERY. Ma trận điểm là (n_query × 184.368); `topk` lại đòi
        bản float32 của nó, nên 5.206 câu một lượt là 3,8 GB cấp phát trong một nhịp —
        đủ để OOM trên máy dùng chung dù nvidia-smi vừa báo còn 8 GB trống. Chia 512
        câu một khối thì đỉnh chỉ còn ~0,4 GB, kết quả không đổi (topk theo từng hàng).
        """
        torch = self.torch
        qs = np.asarray(qvecs, dtype=np.float16)
        k = min(top_k, self.mat.shape[0])
        out_v, out_i = [], []
        for s in range(0, len(qs), block):
            q = torch.from_numpy(qs[s:s + block]).to(self.device)
            if self.device == "cpu":
                q = q.float()
            scores = q @ self.mat.T                  # (block, n_chunk)
            vals, idx = torch.topk(scores.float(), k=k, dim=1)
            out_v.append(vals.cpu().numpy()); out_i.append(idx.cpu().numpy())
            del q, scores, vals, idx
        return np.concatenate(out_v), np.concatenate(out_i)
