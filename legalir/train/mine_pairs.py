"""Sinh dữ liệu huấn luyện reranker từ train.json — không dùng bất kỳ nguồn ngoài nào.

Vấn đề: train.json chỉ cho `document_id`, mà cross-encoder cần biết CHUNK nào trong
văn bản đó mới trả lời câu hỏi. Một gold doc trung bình có ~21 chunk, đa số không liên quan.

Positive (nhãn yếu): trong số các chunk của gold doc, lấy chunk mà tầng retrieval cho
điểm cao nhất. Tính trên TOÀN BỘ chunk của gold doc chứ không chỉ chunk lọt pool, nên
phủ 100% câu kể cả khi retrieval trượt.

Negative (bán khó): lấy từ hạng 10–50 của chính danh sách RRF mà reranker sẽ gặp lúc
inference.
  - không lấy top-10: dày đặc false negative (NV-Retriever đo ~70% 'hard negative'
    trên MS-MARCO thực ra là positive chưa gán nhãn; corpus luật còn tệ hơn vì nhiều
    văn bản gần trùng nội dung).
  - không lấy ngẫu nhiên: quá dễ, và lệch khỏi phân phối lúc inference.
"""
# Port từ training_logs/original_scripts/mine_training_pairs.py. Khác biệt DUY NHẤT
# so với bản gốc (logic/hyperparam giữ nguyên từng dòng):
#   - bỏ `sys.path` hack; `BM25Csr/DenseMatrix/_top_k_indices` ->
#     `BM25Index/DenseIndex/top_k_indices` của legalir.index (đã xác minh cùng interface
#     `.chunk_ids`, `.get_scores(tokens)`, `.mat`, `.device` và cùng spec top-k);
#   - `BGEM3Encoder.encode` -> `encode_queries_bgem3` (cùng chữ ký (dense, idx, val),
#     tham số `type` bị bỏ qua ở cả hai nơi);
#   - `E5Encoder.encode` (có gắn tiền tố theo flag) KHÔNG dùng được
#     `encode_queries_st` (hàm này cố tình không tiền tố cho inference) nên mirror đúng
#     hành vi gốc bằng helper `_encode_e5` local bên dưới (guard pooling + prefix + ST);
#   - thêm `--bge_prefix/--e5_prefix` (ma trận trong gói đặt tên theo checkpoint, không
#     còn tên legacy `bge_dense`/`e5_dense`; bỏ trống thì fallback đúng tên legacy);
#   - default path chuyển sang layout gói (driver luôn truyền explicit);
#   - default `--e5_model` gốc (`intfloat/multilingual-e5-large`) là default chết —
#     chưa từng dùng trong bất kỳ lần chạy nào được ghi nhận (mọi job đều truyền
#     harrier-family explicit; pipeline này không có model họ e5). Đổi thành harrier
#     base cho khỏi bẫy người chạy tay; driver luôn override explicit.
import argparse
import gzip
import json
import os
import time
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

from legalir.encoders import encode_queries_bgem3
from legalir.index.bm25 import BM25Index
from legalir.index.dense import DenseIndex
from legalir.index.topk import top_k_indices


def _encode_e5(model_path, texts, use_prefix=True, batch_size=64):
    """Mirror đúng `retrieval.encoder.E5Encoder.encode` (repo dev).

    E5Encoder là wrapper generic quanh SentenceTransformer: gắn tiền tố "{type}: "
    khi use_prefix=True (default của class, và mine script gốc không bao giờ đổi),
    guard pooling modules.json (lỗi §65), in pooling mode ra log. Không dùng
    `encode_queries_st` vì hàm đó cố tình KHÔNG tiền tố (đúng cho inference harrier).
    """
    import torch
    from sentence_transformers import SentenceTransformer
    if os.path.isdir(model_path) and not os.path.exists(os.path.join(model_path, "modules.json")):
        if os.environ.get("E5_ALLOW_NO_POOLING_CFG") != "1":
            raise SystemExit(
                f"⛔ {model_path}: thiếu modules.json/1_Pooling — SentenceTransformer "
                f"sẽ âm thầm dùng MEAN pooling cho câu hỏi. Vá: chép modules.json, "
                f"1_Pooling/, config_sentence_transformers.json, sentence_bert_config.json "
                f"từ models/harrier_ft_ep2 (last-token). "
                f"Escape có chủ đích: E5_ALLOW_NO_POOLING_CFG=1.")
    model = SentenceTransformer(model_path,
                                device="cuda" if torch.cuda.is_available() else "cpu")
    for m in model:
        if m.__class__.__name__ == "Pooling":
            mode = ("cls" if getattr(m, "pooling_mode_cls_token", False) else
                    "last_token" if getattr(m, "pooling_mode_lasttoken", False) else
                    "mean" if getattr(m, "pooling_mode_mean_tokens", False) else
                    "max" if getattr(m, "pooling_mode_max_tokens", False) else "?")
            print(f"E5Encoder: {model_path} pooling_cau_hoi={mode}")
    out = []
    for i in range(0, len(texts), batch_size):
        part = ([f"query: " + t for t in texts[i:i + batch_size]]
                if use_prefix else texts[i:i + batch_size])
        out.extend(model.encode(part, normalize_embeddings=True))
    del model
    torch.cuda.empty_cache()
    return out


def main(a):
    rng = np.random.default_rng(a.seed)
    data = json.load(open(a.train_file, encoding="utf-8"))
    qids = sorted(data)
    if a.num_shards > 1:
        qids = [q for k, q in enumerate(qids) if k % a.num_shards == a.shard]
    questions = [data[q]["question"] for q in qids]
    gold = {q: set(map(str, data[q]["answer"])) for q in qids}
    print(f"{len(qids)} câu")

    idx_dir = a.index_dir
    bm25 = BM25Index(idx_dir)
    bge_prefix = a.bge_prefix or os.path.join(idx_dir, "bge_dense")
    e5_prefix = a.e5_prefix or os.path.join(idx_dir, "e5_dense")
    with open(bge_prefix + ".ids.json", encoding="utf-8") as f:
        meta = json.load(f)
    chunk_ids, aids = meta["chunk_ids"], np.array(meta["aids"])
    doc_chunks = defaultdict(list)
    for i, ad in enumerate(aids):
        doc_chunks[ad].append(i)
    doc_chunks = {k: np.array(v) for k, v in doc_chunks.items()}
    bm_pos = {c: i for i, c in enumerate(bm25.chunk_ids)}
    sparse_mat = sp.load_npz(os.path.join(idx_dir, "bge_sparse.npz")).T.tocsr()

    texts = {}
    reranker = None
    if a.rerank_model:
        import gzip
        with gzip.open(a.texts, "rt", encoding="utf-8") as f:
            texts = json.load(f)
        from FlagEmbedding import FlagReranker
        reranker = FlagReranker(a.rerank_model, use_fp16=True)
        print(f"vòng 2: đào negative bằng {a.rerank_model}")

    import torch
    t0 = time.time()
    bge_q, sidx, sval = encode_queries_bgem3(a.bge_model, questions, 64)
    torch.cuda.empty_cache()
    e5_q = _encode_e5(a.e5_model, questions, use_prefix=not a.e5_no_prefix)
    torch.cuda.empty_cache()
    print(f"encode {time.time()-t0:.0f}s")

    bge_dm = DenseIndex(bge_prefix)
    e5_dm = DenseIndex(e5_prefix)
    bge_all = torch.from_numpy(np.asarray(bge_q, dtype=np.float16)).to(bge_dm.device)
    e5_all = torch.from_numpy(np.asarray(e5_q, dtype=np.float16)).to(e5_dm.device)

    vocab_n = sparse_mat.shape[0]
    out = open(a.out_file, "w", encoding="utf-8")
    n_written = n_nopos = 0
    B = 32
    t0 = time.time()
    for s in range(0, len(qids), B):
        e = min(s + B, len(qids))
        sb = (bge_all[s:e].float() @ bge_dm.mat.T.float()).cpu().numpy()
        se = (e5_all[s:e].float() @ e5_dm.mat.T.float()).cpu().numpy()
        rows, cols, vals = [], [], []
        for j in range(s, e):
            for x, y in zip(sidx[j], sval[j]):
                x = int(x)
                if x < vocab_n:
                    rows.append(j - s); cols.append(x); vals.append(float(y))
        qm = sp.csr_matrix((np.asarray(vals, dtype=np.float32),
                            (np.asarray(rows), np.asarray(cols))), shape=(e - s, vocab_n))
        ss = (qm @ sparse_mat).toarray()

        for r in range(e - s):
            i = s + r
            qid = qids[i]
            g = gold[qid]
            comb = sb[r] + se[r]                       # điểm gộp để chọn positive

            pos = []
            for d in g:
                ci = doc_chunks.get(d)
                if ci is None or len(ci) == 0:
                    continue
                pos.append(chunk_ids[int(ci[np.argmax(comb[ci])])])
            if not pos:
                n_nopos += 1
                continue

            gold_pool = []
            if reranker is not None:
                for d in g:
                    ci = doc_chunks.get(d)
                    if ci is not None:
                        best = ci[np.argsort(-comb[ci])[:a.gold_topk]]
                        gold_pool += [chunk_ids[int(x)] for x in best]

            bm_scores = bm25.get_scores(questions[i].split())
            lists = [
                [chunk_ids[x] for x in top_k_indices(sb[r], a.depth)],
                [chunk_ids[x] for x in top_k_indices(se[r], a.depth)],
                [chunk_ids[x] for x in top_k_indices(ss[r], a.depth)],
                [bm25.chunk_ids[x] for x in top_k_indices(bm_scores, a.depth_bm25)],
            ]
            fused = defaultdict(float)
            for lst in lists:
                for rk, c in enumerate(lst):
                    fused[c] += 1.0 / (60 + rk)
            order = sorted(fused, key=fused.get, reverse=True)
            aid_of = {c: aids[bm_pos[c]] if c in bm_pos else None for c in order}
            cand = [c for c in order if aid_of.get(c) not in g]

            if reranker is not None:
                # VÒNG 2: xếp lại ứng viên bằng chính model đã fine-tune. Negative
                # hạng 10-50 của retriever zero-shot giờ đã quá dễ với model mới;
                # phải lấy đúng chỗ model HIỆN TẠI còn nhầm thì mới học được thêm.
                head = cand[:a.rerank_depth]
                pairs = [[questions[i], texts[c]] for c in head if c in texts]
                head = [c for c in head if c in texts]
                if head:
                    sc_r = reranker.compute_score(pairs, batch_size=64, max_length=512,
                                                  normalize=True)
                    if not isinstance(sc_r, list):
                        sc_r = [sc_r]
                    head = [c for _, c in sorted(zip(sc_r, head), key=lambda x: -x[0])]
                    cand = head + [c for c in cand if c not in set(head)]
                if gold_pool:
                    gp = [c for c in gold_pool if c in texts]
                    if gp:
                        sg = reranker.compute_score([[questions[i], texts[c]] for c in gp],
                                                    batch_size=64, max_length=512, normalize=True)
                        if not isinstance(sg, list):
                            sg = [sg]
                        best_by_doc = {}
                        for c, v in zip(gp, sg):
                            ad = aids[bm_pos[c]] if c in bm_pos else None
                            if ad is None:
                                continue
                            if ad not in best_by_doc or v > best_by_doc[ad][1]:
                                best_by_doc[ad] = (c, v)
                        if best_by_doc:
                            pos = [v[0] for v in best_by_doc.values()]

            band = cand[a.neg_from:a.neg_to] or cand
            k = min(a.n_neg, len(band))
            negs = list(rng.choice(band, size=k, replace=False)) if k else []
            if len(negs) < a.n_neg:                     # bù từ ngoài dải nếu thiếu
                extra = [c for c in cand if c not in negs][:a.n_neg - len(negs)]
                negs += extra
            if not negs:
                continue
            out.write(json.dumps({"qid": qid, "query": questions[i],
                                  "pos": pos, "neg": [str(x) for x in negs]},
                                 ensure_ascii=False) + "\n")
            n_written += 1
        if s % (B * 20) == 0:
            done = e
            print(f"  {done}/{len(qids)} · {(time.time()-t0)/max(done,1)*1000:.0f} ms/câu", flush=True)
    out.close()
    print(f"\n✅ {a.out_file} · {n_written} mẫu · {n_nopos} câu không tìm được positive")


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="legalir.train.mine_pairs")
    p.add_argument("--train_file", default="splits_v2/train.json")
    p.add_argument("--out_file", default="train_pairs/legalir_pairs.jsonl")
    p.add_argument("--index_dir", default="index/db_fast")
    p.add_argument("--bge_model", default="models/bge-m3")
    p.add_argument("--e5_model", default="intfloat/multilingual-e5-large",
                   help="GIỮ NGUYÊN default gốc (stale — chưa từng dùng trong bất kỳ lần "
                        "chạy nào được ghi nhận; mọi job đều truyền harrier-family explicit, "
                        "pipeline này không có model họ e5). Driver luôn override explicit.")
    p.add_argument("--e5_no_prefix", action="store_true",
                   help="tắt tiền tố 'query: ' của E5Encoder gốc (mặc định BẬT như bản gốc)")
    p.add_argument("--bge_prefix", default=None,
                   help="prefix ma trận dense kênh bge (mặc định: <index_dir>/bge_dense legacy)")
    p.add_argument("--e5_prefix", default=None,
                   help="prefix ma trận dense kênh e5 (mặc định: <index_dir>/e5_dense legacy)")
    p.add_argument("--depth", type=int, default=50)
    p.add_argument("--depth_bm25", type=int, default=50)
    p.add_argument("--neg_from", type=int, default=10)
    p.add_argument("--neg_to", type=int, default=50)
    p.add_argument("--n_neg", type=int, default=7)
    p.add_argument("--rerank_model", default="",
                   help="model dùng xếp lại ứng viên khi đào negative vòng 2")
    p.add_argument("--texts", default="index/db_fast/texts.json.gz")
    p.add_argument("--rerank_depth", type=int, default=40)
    p.add_argument("--gold_topk", type=int, default=6)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out_file), exist_ok=True)
    main(a)
