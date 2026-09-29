"""Pipeline LegalIR đã tối ưu — cùng thuật toán baseline, khác cách thực thi.

Khác biệt so với retrieval/run_pipeline.py, mỗi thứ đều có số đo trong result.md:

  1. sparse của bge-m3 chạy bằng CSR, không qua Qdrant     15.000 -> 10 ms/câu
  2. BM25 chạy bằng ma trận trọng số, không qua rank_bm25    2.350 -> 30 ms/câu
  3. dense chạy bằng matmul trên GPU, không qua Qdrant       2.070 ->  8 ms/câu
  4. bỏ kênh dense thứ ba (gte)          recall@5 không đổi, -305M tham số
  5. bm25 top-200 -> top-50              pool 239 -> 93 chunk, recall@5 không đổi
  6. reranker Vietnamese_Reranker        +3,8 điểm chunk_hit@1 cho Task 2

Cần chạy retrieval/fast_index.py để dựng db_fast trước (xem README ở result.md §8).

Tách hai bước như run_pipeline.py, ghi ĐÚNG định dạng pool của eval/:

    chan_{bm25,bge,e5,bgesparse}.jsonl.gz   mỗi dòng {"qid","question","rows":[[cid,aid,score]]}
    texts_fast.json.gz                       cid -> text, chỉ chunk nằm trong pool
    timing_fast.json

Nhờ vậy eval/eval_rerank.py, ablation_channels.py, compare_rerankers.py, paired_test.py
dùng lại được pool này — đổi reranker không phải chạy lại tầng retrieval.

⚠️ Chuỗi PRODUCTION hiện tại (public 0,9554 — submission_qwen3rr_headings_a0.15) KHÔNG
đi qua step2 của file này: step1 dựng pool → eval/score_pairs.py chấm CE `fam` →
eval/build_qwen3_submission.py gộp LSE T=2 + pha điểm đề mục Qwen3-Reranker α=0,15.
Xem docs/ARCH_submission_qwen3rr_headings_a0.15.md. step2 ở đây tái lập hai bài nộp cũ
(max-agg 0,9471 · LSE T=2 0,9519) làm đối chứng — cổng eval/verify_agg_parity.py.

    python -m retrieval.run_pipeline_fast --mode step1 --save_dir eval/pools/dev_fast ...
    python -m retrieval.run_pipeline_fast --mode step2 --load_dir eval/pools/dev_fast ...
    python -m retrieval.run_pipeline_fast --mode full  --save_dir ... (step1 rồi step2)
"""
import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.fast_index import BM25Csr, DenseMatrix  # noqa: E402

CHANNELS = ["bm25", "bge", "e5", "bgesparse"]


def _load_questions(path):
    qdata = json.load(open(path, encoding="utf-8"))
    if isinstance(qdata, list):
        qdata = {str(it.get("id", i)): it for i, it in enumerate(qdata)}
    qids = sorted(qdata)
    return qids, [qdata[q]["question"] for q in qids]


def _check_chunk_universe(ref_ids, others):
    """Bốn kênh phải nói về CÙNG một tập chunk.

    aid (document_id) chỉ có trong file id của kênh bge; các kênh khác tra nhờ bảng đó.
    Nếu tập chunk lệch nhau thì candidate của bm25/e5/sparse sẽ tra trượt và bị bỏ —
    im lặng, sau khi đã tốn tiền chạy qua reranker, recall tụt mà không ai biết.

    Lệch được là vì prepare_data/build_db_corpus_dsc.py ingest 3 collection bằng 3 vòng
    resume độc lập theo points_count của riêng từng collection: một lần OOM giữa chừng ở
    e5 trong khi bge đã xong là đủ. Nên kiểm một lần ở đây, hỏng thì hỏng ồn ào.
    """
    ref = set(ref_ids)
    for name, ids in others:
        missing = set(ids) - ref
        if missing:
            raise SystemExit(
                f"Kênh '{name}' có {len(missing)} chunk không nằm trong tập của kênh bge "
                f"(vd: {sorted(missing)[:3]}). Index bị lệch — dựng lại db_fast trước khi chạy.")


def run_step1(args):
    t_start = time.time()
    qids, questions = _load_questions(args.question_file)
    print(f"{len(qids)} câu hỏi")

    print("nạp index ...")
    t = time.time()
    # --bm25_index_dir: cho phep kenh BM25 doc index KHAC voi ba kenh kia. Dung de thu
    # BM25 tren corpus DA TACH TU trong khi dense/sparse van dung corpus goc — doi mot
    # bien mot lan. Tap chunk phai trung nhau, _check_chunk_universe se bat neu lech.
    bm25 = BM25Csr(args.bm25_index_dir or args.index_dir)
    sparse_mat = sp.load_npz(os.path.join(args.index_dir, "bge_sparse.npz")).T.tocsr()
    with gzip.open(os.path.join(args.index_dir, "bge_sparse.npz.ids.json.gz"),
                   "rt", encoding="utf-8") as f:
        sp_meta = json.load(f)
    print(f"  {time.time()-t:.1f}s")

    # ---- encode query -----------------------------------------------------
    from retrieval.encoder import BGEM3Encoder, E5Encoder
    import torch
    t = time.time()
    bge_enc = BGEM3Encoder(model_name=args.bge_model, use_fp16=True)
    bge_dense_q, sp_idx, sp_val = [], [], []
    for i in range(0, len(questions), args.batch_size):
        d, ix, vl = bge_enc.encode(questions[i:i + args.batch_size], type="query")
        bge_dense_q.extend(d); sp_idx.extend(ix); sp_val.extend(vl)
    del bge_enc
    torch.cuda.empty_cache()

    # --bge_dense_model: dùng bge-m3 ĐÃ FINE-TUNE cho kênh dense, trong khi kênh sparse
    # vẫn phải do model gốc sinh (bản fine-tune là AutoModel thuần, không có head sparse).
    # Phải encode query đúng như encode_corpus.py đã encode document: CLS pooling,
    # normalize, không tiền tố — sai một trong ba thứ đó là embedding lệch hệ toạ độ.
    if args.bge_dense_model:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.bge_dense_model)
        mdl = AutoModel.from_pretrained(args.bge_dense_model, add_pooling_layer=False)
        mdl = mdl.half().to("cuda" if torch.cuda.is_available() else "cpu").eval()
        bge_dense_q = []
        with torch.inference_mode():
            for i in range(0, len(questions), args.batch_size):
                enc = tok(questions[i:i + args.batch_size], padding=True, truncation=True,
                          max_length=args.q_len, return_tensors="pt").to(mdl.device)
                h = mdl(**enc).last_hidden_state[:, 0]          # CLS
                h = torch.nn.functional.normalize(h.float(), dim=-1)
                bge_dense_q.extend(h.cpu().numpy().astype(np.float16))
        del mdl; torch.cuda.empty_cache()

    e5_enc = E5Encoder(model_name=args.e5_model,
                       use_prefix=not args.e5_no_prefix,
                       segment=args.e5_segment_query)
    e5_q = []
    for i in range(0, len(questions), args.batch_size):
        d, _, _ = e5_enc.encode(questions[i:i + args.batch_size], type="query")
        e5_q.extend(d)
    del e5_enc; torch.cuda.empty_cache()
    t_encode = time.time() - t

    # Giữ chỗ VRAM cho phần tốn nhất còn lại (ma trận dense + matmul + topk). PHẢI đặt
    # ở ĐÂY: mọi lời gọi from_pretrained ở trên đều gọi empty_cache qua accelerate và
    # xoá sạch chỗ đã giữ. Xem eval/gpu_reserve.py.
    if args.reserve_gb > 0:
        from eval.gpu_reserve import reserve_vram
        reserve_vram(args.reserve_gb, 0, ballast_gb=args.ballast_gb)

    def _drop():
        """empty_cache() TRẢ bộ nhớ về driver — tức là tự tay huỷ chỗ vừa giữ.
        Đã đo: log báo giữ 5,00 GB nhưng nvidia-smi chỉ thấy 212 MiB, vì hai lời gọi
        empty_cache() ở dưới (dọn ma trận dense giữa hai kênh) xoá sạch. Khi đang giữ
        chỗ thì `del` là đủ: bộ nhớ về cache của CHÍNH TA và được tái dùng ngay."""
        if args.reserve_gb <= 0:
            torch.cuda.empty_cache()

    # ---- 4 kênh -----------------------------------------------------------
    t = time.time()
    bge_dm = DenseMatrix(args.bge_dense_prefix
                         or os.path.join(args.index_dir, "bge_dense"))
    bge_val, bge_idx = bge_dm.search_batch(np.asarray(bge_dense_q), args.top_k_dense)
    bge_ids, bge_aids = bge_dm.chunk_ids, bge_dm.aids
    del bge_dm; _drop()

    # --e5_dense_prefix cho phép trỏ sang ma trận của e5 ĐÃ FINE-TUNE. Vector của
    # index cũ do model cũ sinh ra, không so sánh được với query vector của model mới,
    # nên đổi model thì BẮT BUỘC đổi luôn ma trận — hai cờ này phải đi cùng nhau.
    e5_prefix = args.e5_dense_prefix or os.path.join(args.index_dir, "e5_dense")
    e5_dm = DenseMatrix(e5_prefix)
    e5_val, e5_idx = e5_dm.search_batch(np.asarray(e5_q), args.top_k_dense)
    e5_ids = e5_dm.chunk_ids
    del e5_dm; _drop()

    # --extra_dense "ten|model|prefix_ma_tran|pooling|tien_to_query": THEM mot kenh dense
    # nua thay vi THAY kenh e5. Ghi chu dau file noi kenh dense thu ba (gte) tung bi bo
    # vi "recall@5 khong doi" — nhung do la do o cau hinh zero-shot cu; gio ba tang kia
    # da khac han, va vnembv2/vnemb1 cham sat nut e5-ft (0,9424 / 0,9423 vs 0,9428) nen
    # dang do lai. Luu y day KHONG phai RRF co trong so: kien truc nay lay HOP cac kenh
    # lam pool roi de reranker cham lai tat ca, nen them kenh = pool rong hon. Pool rong
    # hon tung lam TE di 0,28 diem (them document theo do thi trich dan) — reranker phai
    # chon giua nhieu nhieu hon. Vi the phai do chu khong duoc doan.
    extra_ids = {}
    for spec in (args.extra_dense or []):
        nm, mdl_path, pfx, pool_mode, qpfx = (spec.split("|") + ["", "", "", ""])[:5]
        from transformers import AutoModel, AutoTokenizer
        tk = AutoTokenizer.from_pretrained(mdl_path)
        md = AutoModel.from_pretrained(mdl_path)
        md = md.half().to("cuda" if torch.cuda.is_available() else "cpu").eval()
        qv = []
        with torch.inference_mode():
            for i in range(0, len(questions), args.batch_size):
                batch = [qpfx + q for q in questions[i:i + args.batch_size]]
                e = tk(batch, padding=True, truncation=True, max_length=args.q_len,
                       return_tensors="pt").to(md.device)
                h = md(**e).last_hidden_state
                if pool_mode == "cls":
                    v = h[:, 0]
                else:
                    m = e["attention_mask"].unsqueeze(-1).float()
                    v = (h * m).sum(1) / m.sum(1)
                v = torch.nn.functional.normalize(v.float(), dim=-1)
                qv.extend(v.cpu().numpy().astype(np.float16))
        del md; torch.cuda.empty_cache()
        dm = DenseMatrix(pfx)
        val, idx = dm.search_batch(np.asarray(qv), args.top_k_dense)
        extra_ids[nm] = (dm.chunk_ids, val, idx)
        del dm; _drop()

    # aid chỉ có ở file id của kênh bge -> bắt buộc mọi kênh cùng một tập chunk
    _check_chunk_universe(bge_ids, [("bm25", bm25.chunk_ids),
                                    ("e5", e5_ids),
                                    ("bgesparse", sp_meta["chunk_ids"])]
                                   + [(k, v[0]) for k, v in extra_ids.items()])
    aid_of = dict(zip(bge_ids, bge_aids))

    # Cau hoi dua vao BM25 phai duoc tach tu DUNG NHU corpus da tach — neu corpus co
    # "hop_dong" ma cau hoi van la "hop dong" thi khong con token nao khop, BM25 ve 0.
    if args.bm25_segment:
        from underthesea import word_tokenize
        bm25_questions = [word_tokenize(q, format="text") for q in questions]
    else:
        bm25_questions = questions

    vocab_n = sparse_mat.shape[0]
    rows, cols, vals = [], [], []
    for j, (ix, vl) in enumerate(zip(sp_idx, sp_val)):
        for a, bv in zip(ix, vl):
            a = int(a)
            if a < vocab_n:
                rows.append(j); cols.append(a); vals.append(float(bv))
    qmat = sp.csr_matrix((np.asarray(vals, dtype=np.float32),
                          (np.asarray(rows), np.asarray(cols))),
                         shape=(len(questions), vocab_n))

    def _row(cid, score):
        aid = aid_of.get(cid)
        if aid is None:                      # _check_chunk_universe đã loại khả năng này
            raise SystemExit(f"chunk {cid!r} không có document_id — index lệch")
        return [cid, str(aid), float(score)]

    all_chans = CHANNELS + list(extra_ids)
    chans = {c: {} for c in all_chans}
    pools = []
    for s in range(0, len(qids), 64):
        block = (qmat[s:s + 64] @ sparse_mat).toarray()
        for r in range(block.shape[0]):
            i = s + r
            qid = qids[i]
            sc = block[r]
            k = min(args.top_k_dense, sc.size)
            top = np.argpartition(-sc, k - 1)[:k]
            top = top[np.argsort(-sc[top])]
            chans["bgesparse"][qid] = [_row(sp_meta["chunk_ids"][x], sc[x])
                                       for x in top if sc[x] > 0]
            chans["bm25"][qid] = [_row(c, _s)
                                  for _s, c in bm25.search(bm25_questions[i],
                                                           top_k=args.top_k_bm25)]
            chans["bge"][qid] = [_row(bge_ids[x], v)
                                 for v, x in zip(bge_val[i], bge_idx[i])]
            chans["e5"][qid] = [_row(e5_ids[x], v)
                                for v, x in zip(e5_val[i], e5_idx[i])]
            for nm, (cids, val, idx) in extra_ids.items():
                chans[nm][qid] = [_row(cids[x], v) for v, x in zip(val[i], idx[i])]
            pools.append(sorted({row[0] for c in all_chans for row in chans[c][qid]}))
    t_retrieval = time.time() - t
    avg_pool = sum(len(p) for p in pools) / len(pools)
    print(f"  encode {t_encode:.1f}s · retrieval {t_retrieval:.1f}s · "
          f"pool {avg_pool:.0f} chunk/câu")

    # ---- text cho reranker, chỉ phần nằm trong pool ------------------------
    need = set().union(*[set(p) for p in pools])
    texts = {}
    for path in args.text_source:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for cid, tx in json.load(f).items():
                if cid in need:
                    texts[cid] = tx
    missing = len(need) - len(texts)
    if missing:
        raise SystemExit(f"thiếu text cho {missing} chunk — bổ sung --text_source")

    # ---- ghi pool đúng định dạng eval/ -------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    for name in all_chans:
        p = os.path.join(args.save_dir, f"chan_{name}.jsonl.gz")
        with gzip.open(p, "wt", encoding="utf-8") as f:
            for i, qid in enumerate(qids):
                f.write(json.dumps({"qid": qid, "question": questions[i],
                                    "rows": chans[name][qid]}, ensure_ascii=False) + "\n")
        print(f"  {p}")
    with gzip.open(os.path.join(args.save_dir, "texts_fast.json.gz"),
                   "wt", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    timing = {"encode": t_encode, "retrieval": t_retrieval,
              "total": time.time() - t_start, "n_queries": len(qids),
              "avg_pool": avg_pool}
    with open(os.path.join(args.save_dir, "timing_fast.json"), "w") as f:
        json.dump(timing, f, indent=2)
    print(f"\n✅ step1: {len(need)} chunk trong pool, {timing['total']:.0f}s")


def _logit(p):
    """Điểm CE `normalize=True` nằm trong [0,1]; LSE phải cộng trong không gian LOGIT
    (giống eval/agg_sweep2.logit) chứ không phải trên xác suất."""
    z = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(z / (1 - z))


def _corpus_nchunk(ids_path):
    ids = json.load(open(ids_path, encoding="utf-8"))
    c = {}
    for i in ids:
        d = i.split("_", 1)[0]
        c[d] = c.get(d, 0) + 1
    return c


def _aggregate(pool, chunk_scores, aid_of, agg, T, lam, nchunk, tie, top_k):
    """chunk -> document. Trùng khít eval/agg_sweep_kfold.predict().

    agg=max  : điểm doc = max điểm chunk (hành vi gốc của file này).
    agg=lse  : điểm doc = T·logsumexp(logit(chunk)/T) — thưởng document có NHIỀU chunk
               cùng tốt. Cấu hình bài submission_lse_T2 (T=2, public 0,9519);
               production hiện tại (0,9554) dùng cùng LSE T=2 nhưng pha thêm điểm
               đề mục trong eval/build_qwen3_submission.py — xem
               docs/ARCH_submission_qwen3rr_headings_a0.15.md.
    tie=det  : phá hoà tất định theo (-điểm, document_id) — phép của
               kfold_clean_report.py. §25a: thứ tự chèn làm lệch 0,08 điểm khi hoà
               điểm ở slot 5. tie=stable giữ nguyên hành vi cũ để đối chứng.
    """
    by = {}
    for cid, sc in zip(pool, chunk_scores):
        by.setdefault(aid_of[cid], []).append(sc)

    out = {}
    for d, vs in by.items():
        if agg == "max":
            s = float(max(vs))
        elif agg == "lse":
            lg = _logit(vs)
            m = lg.max()
            s = float(T * (m / T + np.log(np.exp(lg / T - m / T).sum()))) if T > 0 \
                else float(m)
        else:
            raise SystemExit(f"--agg không hợp lệ: {agg}")
        if lam:
            s -= lam * np.log(max(nchunk.get(d, 1), 1))
        out[d] = s

    if tie == "det":
        order = sorted(out, key=lambda d: (-out[d], d))
    else:
        order = sorted(out, key=out.get, reverse=True)
    return order[:top_k]


def _load_pools(load_dir):
    """Đọc lại pool step1 — cùng định dạng eval/eval_rerank.py:load_pools()."""
    import glob
    merged, questions = {}, {}
    files = sorted(glob.glob(os.path.join(load_dir, "chan_*.jsonl.gz")))
    if not files:
        raise SystemExit(f"Không thấy chan_*.jsonl.gz nào trong {load_dir}")
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                questions[d["qid"]] = d["question"]
                merged.setdefault(d["qid"], []).extend(d["rows"])
    texts = {}
    for path in sorted(glob.glob(os.path.join(load_dir, "texts_*.json.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            texts.update(json.load(f))
    print(f"nạp {len(files)} kênh, {len(merged)} câu, {len(texts)} chunk có text")
    return questions, merged, texts


def run_step2(args):
    t_start = time.time()
    if args.agg == "max" and args.tie_break == "stable":
        print("ℹ️  step2 đang chạy DEFAULT (max-agg + tie stable) = cấu hình cũ, public "
              "0,9471. Bài LSE 0,9519: --agg lse --agg_T 2 --tie_break det. Production "
              "0,9554 không đi qua step2 — docs/ARCH_submission_qwen3rr_headings_a0.15.md.")
    questions, merged, texts = _load_pools(args.load_dir)
    qids = sorted(merged)

    # Gộp candidate 4 kênh, bỏ trùng chunk TRONG TỪNG CÂU, giữ aid đi kèm.
    # aid_of dùng chung cho mọi câu (chunk -> document là cố định), nhưng pool thì
    # phải tính riêng từng câu — lấy aid_of làm bộ lọc trùng sẽ khiến chunk đã xuất
    # hiện ở câu trước bị loại khỏi câu sau, pool rỗng dần rồi ra 0 document.
    pools, aid_of = [], {}
    for qid in qids:
        pool = set()
        for cid, aid, _score in merged[qid]:
            pool.add(cid)
            aid_of[cid] = aid
        pools.append(sorted(pool))

    empty = [qids[i] for i, p in enumerate(pools) if not p]
    if empty:
        raise SystemExit(f"{len(empty)} câu có pool rỗng sau khi gộp 4 kênh: {empty[:5]}")

    lack = [c for p in pools for c in p if c not in texts]
    if lack:
        raise SystemExit(f"thiếu text cho {len(lack)} chunk trong pool")

    t = time.time()
    from FlagEmbedding import FlagReranker
    model = FlagReranker(args.reranker, use_fp16=True)
    if args.reserve_gb > 0:
        # Giu cho VRAM ngay sau khi reranker len GPU (dat truoc do se bi
        # from_pretrained goi empty_cache xoa sach). Chong bi chen giua chung tren
        # may dung chung. Xem eval/gpu_reserve.py.
        from eval.gpu_reserve import reserve_vram
        reserve_vram(args.reserve_gb, 0, ballast_gb=args.ballast_gb)
    flat, spans = [], []
    for i, pool in enumerate(pools):
        start = len(flat)
        flat.extend([[questions[qids[i]], texts[c]] for c in pool])
        spans.append((start, len(flat)))
    scores = model.compute_score(flat, batch_size=args.batch_size_rerank,
                                 max_length=args.max_length, normalize=True)

    # --reranker2: chạy NỐI TIẾP rồi trung bình điểm. Nối tiếp chứ không song song vì
    # hai model cùng cỡ 568M, giữ cả hai trên GPU là gấp đôi VRAM mà không nhanh hơn —
    # rerank vốn đã bão hoà compute. `normalize=True` đưa cả hai về [0,1] nên cộng
    # trung bình có nghĩa; trên dev v2 mean thắng RRF (0,9444 vs 0,9431).
    if args.reranker2:
        del model
        import torch
        # Phải empty_cache thật ở đây (nhường chỗ cho model thứ hai nạp vào), nên chỗ
        # đã giữ mất theo — khoá LẠI ngay sau khi model2 lên GPU.
        torch.cuda.empty_cache()
        model2 = FlagReranker(args.reranker2, use_fp16=True)
        if args.reserve_gb > 0:
            reserve_vram(args.reserve_gb, 0, ballast_gb=args.ballast_gb)
        scores2 = model2.compute_score(flat, batch_size=args.batch_size_rerank,
                                       max_length=args.max_length, normalize=True)
        w = args.reranker_weight
        scores = [w * a + (1 - w) * b for a, b in zip(scores, scores2)]
    t_rerank = time.time() - t

    # ---- gộp chunk -> doc, cắt đúng 5 --------------------------------------
    # `--agg lse` tái lập ĐÚNG eval/build_lse_submission.py (bài submission_lse_T2,
    # public 0,9519; production 0,9554 = LSE + đề mục qua build_qwen3_submission.py).
    # Trước 08/09 file này chỉ có `max` (public 0,9471) trong khi
    # CLAUDE.md §3.1 gọi nó là "pipeline inference chính" ⇒ ai tái lập theo tài liệu
    # ra thấp hơn bài nộp 0,48 điểm. Xem research_paper §3c.6 mục C.
    nchunk = _corpus_nchunk(args.corpus_ids) if args.agg_lam else {}
    preds = {}
    for i, qid in enumerate(qids):
        s, e = spans[i]
        top = _aggregate(pools[i], scores[s:e], aid_of, args.agg, args.agg_T,
                         args.agg_lam, nchunk, args.tie_break, args.top_k_final)
        preds[qid] = {"answer": [str(d) for d in top]}

    # ---- cổng chặn Ô TRỐNG RECALL ------------------------------------------
    # Recall + trần cứng 5 doc: trả ít hơn 5 CHỈ CÓ THỂ mất điểm, không bao giờ được
    # điểm (doc thừa sai không bị Recall phạt). Trước 08/09 code cắt [:5] rồi im lặng
    # đi tiếp nếu pool có <5 document — đã xảy ra thật ở
    # submission_depthtest_rebalance.json (1 câu trả 4 doc).
    short = [q for q, v in preds.items() if len(v["answer"]) < args.top_k_final]
    if short:
        msg = (f"⚠️  {len(short)}/{len(qids)} câu có DƯỚI {args.top_k_final} document "
               f"trong pool ⇒ bỏ trống slot recall. Tăng --top_k_bm25/--top_k_dense. "
               f"Ví dụ: {short[:5]}")
        if args.require_full5:
            raise SystemExit(msg.replace("⚠️ ", "⛔"))
        print(msg)

    # KHÔNG dùng assert: python -O xoá sạch assert, ràng buộc này mà mất thì câu sai
    # lọt thẳng vào submission (0 điểm câu đó, hoặc 0 điểm toàn bài nếu lệch tập khoá).
    bad = [q for q, v in preds.items() if not 1 <= len(v["answer"]) <= 5]
    if bad:
        raise SystemExit(
            f"{len(bad)} câu có số document ngoài [1,5] — scorer sẽ cho 0 điểm. "
            f"Ví dụ: {bad[:5]}")
    dup = [q for q, v in preds.items() if len(set(v["answer"])) != len(v["answer"])]
    if dup:
        raise SystemExit(f"{len(dup)} câu có document_id trùng nhau. Ví dụ: {dup[:5]}")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    n = len(qids)
    total = time.time() - t_start
    print(f"\n✅ {args.output_file}")
    print(f"   rerank {t_rerank/n*1000:6.0f} ms/câu · step2 {total/n*1000:6.0f} ms/câu "
          f"({total/60:.1f} phút)")

    if args.gt_file:
        import importlib.util
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            "btc", os.path.join(root, "scoring", "legalir", "scoring.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        gt = {k: [str(x) for x in v]
              for k, v in json.load(open(args.gt_file, encoding="utf-8")).items()}
        gt = {k: v for k, v in gt.items() if k in preds}
        print("   scorer BTC:", mod.eval_retrieval({k: preds[k] for k in gt}, gt))


def main(args):
    if args.mode in ("step1", "full"):
        run_step1(args)
    if args.mode in ("step2", "full"):
        if args.mode == "full":
            args.load_dir = args.save_dir
        run_step2(args)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["full", "step1", "step2"], default="full",
                   help="full: cả hai; step1: encode+retrieval; step2: rerank")
    p.add_argument("--question_file", required=True)
    p.add_argument("--output_file", default="submission.json")
    p.add_argument("--gt_file", default="")
    p.add_argument("--index_dir", default="retrieval/db_fast")
    p.add_argument("--save_dir", default="retrieval/output/step1_fast")
    p.add_argument("--load_dir", default="retrieval/output/step1_fast")
    p.add_argument("--text_source", nargs="+",
                   default=["retrieval/db_fast/texts.json.gz"])
    p.add_argument("--bge_model", default="BAAI/bge-m3")
    p.add_argument("--bge_dense_model", default="",
                   help="checkpoint bge-m3 đã fine-tune, chỉ dùng cho kênh DENSE "
                        "(sparse vẫn do --bge_model gốc sinh). Đi kèm bắt buộc với "
                        "--bge_dense_prefix trỏ tới ma trận do chính nó encode.")
    p.add_argument("--bge_dense_prefix", default="")
    p.add_argument("--q_len", type=int, default=64,
                   help="độ dài query khi encode bằng --bge_dense_model, phải khớp "
                        "lúc train (train_biencoder.py --q_len)")
    p.add_argument("--e5_model", default="intfloat/multilingual-e5-large")
    p.add_argument("--e5_no_prefix", action="store_true",
                   help="tat tien to query:/passage: — bat cho model nen PhoBERT")
    p.add_argument("--e5_segment_query", action="store_true",
                   help="tach tu cau hoi truoc khi encode — bat cho model nen PhoBERT")
    p.add_argument("--bm25_index_dir", default="",
                   help="index rieng cho kenh BM25 (vd corpus da tach tu); "
                        "mac dinh dung chung --index_dir")
    p.add_argument("--bm25_segment", action="store_true",
                   help="tach tu cau hoi bang underthesea truoc khi dua vao BM25 — "
                        "BAT BUOC bat khi --bm25_index_dir tro toi index da tach tu "
                        "(hướng corpus tách từ ĐÃ ĐÓNG — CLAUDE.md §5; giữ để tái lập)")
    p.add_argument("--extra_dense", nargs="+", default=[],
                   help="ten|model|prefix_ma_tran|cls hoac mean|tien_to_query — "
                        "THEM kenh dense, ghi ra chan_<ten>.jsonl.gz; step2 tu gop "
                        "vi _load_pools duyet glob chan_*.jsonl.gz "
                        "(hướng kênh thứ 5 ĐÃ ĐÓNG — result.md §60; giữ để tái lập)")
    p.add_argument("--e5_dense_prefix", default="",
                   help="ma trận dense của e5; đổi cùng lúc với --e5_model")
    p.add_argument("--reranker", default="AITeamVN/Vietnamese_Reranker")
    p.add_argument("--reranker2", default="",
                   help="reranker thứ hai, chạy nối tiếp rồi trung bình điểm "
                        "(hướng ensemble ĐÃ ĐÓNG — result.md §10/§23a/§62; giữ để tái lập)")
    p.add_argument("--reranker_weight", type=float, default=0.5,
                   help="trọng số của --reranker trong trung bình")
    p.add_argument("--top_k_bm25", type=int, default=50)
    p.add_argument("--top_k_dense", type=int, default=30)
    p.add_argument("--top_k_final", type=int, default=5)
    p.add_argument("--agg", choices=["max", "lse"], default="max",
                   help="gộp chunk->document. `lse` = bài submission_lse_T2 "
                        "(--agg lse --agg_T 2 --tie_break det, public 0,9519); "
                        "`max` giữ hành vi cũ (public 0,9471). Production 0,9554 "
                        "không qua step2 — xem docs/ARCH_submission_qwen3rr_headings_a0.15.md")
    p.add_argument("--agg_T", type=float, default=2.0,
                   help="nhiệt độ của logsumexp; chỉ dùng khi --agg lse")
    p.add_argument("--agg_lam", type=float, default=0.0,
                   help="phạt log(số chunk trong corpus). ĐÃ BỊ BÁC 3 lần độc lập "
                        "(result.md §27k) — để 0")
    p.add_argument("--corpus_ids", default="retrieval/db_fast/bm25_ids.json",
                   help="chỉ đọc khi --agg_lam khác 0")
    p.add_argument("--tie_break", choices=["stable", "det"], default="stable",
                   help="`det` phá hoà theo (-điểm, doc_id) — bắt buộc để trùng khít "
                        "eval/build_lse_submission.py")
    p.add_argument("--require_full5", action="store_true",
                   help="bắt lỗi thay vì cảnh báo khi có câu dưới 5 document")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--batch_size_rerank", type=int, default=64)
    p.add_argument("--ballast_gb", type=float, default=0.0,
                   help="phần KHOÁ CỨNG không bao giờ nhả = (khe muốn giữ) − (đỉnh "
                        "thật của mình)")
    p.add_argument("--reserve_gb", type=float, default=0.0,
                   help="giữ trước VRAM tới đỉnh, chống bị chen giữa chừng")
    p.add_argument("--max_length", type=int, default=512)
    main(p.parse_args())
