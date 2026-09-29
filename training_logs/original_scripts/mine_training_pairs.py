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
import argparse, gzip, json, os, sys, time
from collections import defaultdict
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.fast_index import BM25Csr, DenseMatrix
from retrieval.search import _top_k_indices


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
    bm25 = BM25Csr(idx_dir)
    with open(os.path.join(idx_dir, "bge_dense.ids.json"), encoding="utf-8") as f:
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

    from retrieval.encoder import BGEM3Encoder, E5Encoder
    import torch
    t0 = time.time()
    enc = BGEM3Encoder(model_name=a.bge_model, use_fp16=True)
    bge_q, sidx, sval = [], [], []
    for i in range(0, len(questions), 64):
        d, ix, vl = enc.encode(questions[i:i + 64], type="query")
        bge_q.extend(d); sidx.extend(ix); sval.extend(vl)
    del enc; torch.cuda.empty_cache()
    e5e = E5Encoder(model_name=a.e5_model)
    e5_q = []
    for i in range(0, len(questions), 64):
        d, _, _ = e5e.encode(questions[i:i + 64], type="query")
        e5_q.extend(d)
    del e5e; torch.cuda.empty_cache()
    print(f"encode {time.time()-t0:.0f}s")

    bge_dm = DenseMatrix(os.path.join(idx_dir, "bge_dense"))
    e5_dm = DenseMatrix(os.path.join(idx_dir, "e5_dense"))
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
                [chunk_ids[x] for x in _top_k_indices(sb[r], a.depth)],
                [chunk_ids[x] for x in _top_k_indices(se[r], a.depth)],
                [chunk_ids[x] for x in _top_k_indices(ss[r], a.depth)],
                [bm25.chunk_ids[x] for x in _top_k_indices(bm_scores, a.depth_bm25)],
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
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", default="eval/splits_v2/train.json")
    p.add_argument("--out_file", default="eval/train_pairs/legalir_pairs.jsonl")
    p.add_argument("--index_dir", default="retrieval/db_fast")
    p.add_argument("--bge_model", default="BAAI/bge-m3")
    p.add_argument("--e5_model", default="intfloat/multilingual-e5-large")
    p.add_argument("--depth", type=int, default=50)
    p.add_argument("--depth_bm25", type=int, default=50)
    p.add_argument("--neg_from", type=int, default=10)
    p.add_argument("--neg_to", type=int, default=50)
    p.add_argument("--n_neg", type=int, default=7)
    p.add_argument("--rerank_model", default="",
                   help="model dùng xếp lại ứng viên khi đào negative vòng 2")
    p.add_argument("--texts", default="retrieval/db_fast/texts.json.gz")
    p.add_argument("--rerank_depth", type=int, default=40)
    p.add_argument("--gold_topk", type=int, default=6)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out_file), exist_ok=True)
    main(a)
