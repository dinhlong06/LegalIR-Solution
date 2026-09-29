"""Chứng minh index nhanh cho kết quả trùng khít bản gốc, không chỉ 'gần giống'."""
import json, pickle, sys, time, types
sys.path.insert(0, '.')
import numpy as np
import retrieval.search as rs
sys.modules.setdefault("ViDRILL", types.ModuleType("ViDRILL"))
sys.modules.setdefault("ViDRILL.search", rs)
from retrieval.fast_index import BM25Csr

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
qs = [q["question"] for q in list(json.load(open("eval/splits/dev_questions.json")).values())[:N]]

print("load bản gốc (rank_bm25) ...")
old = pickle.load(open("retrieval/db/bm25_index.pkl", "rb"))
print("load bản CSR ...")
t = time.time(); new = BM25Csr("retrieval/db_fast"); print(f"  {time.time()-t:.1f}s")

assert old.index_corpus == new.chunk_ids, "thứ tự chunk lệch nhau"

max_abs = 0.0; rank_mismatch = 0; t_old = t_new = 0.0
for q in qs:
    tok = q.split()
    t0 = time.time(); s_old = old.bm25.get_scores(tok); t_old += time.time() - t0
    t0 = time.time(); s_new = new.get_scores(tok);       t_new += time.time() - t0
    max_abs = max(max_abs, float(np.abs(s_old - s_new).max()))
    for K in (5, 50, 200, 500):
        if list(rs._top_k_indices(s_old, K)) != list(rs._top_k_indices(s_new, K)):
            rank_mismatch += 1

n = len(qs)
print(f"\n{n} câu")
print(f"  sai số điểm tuyệt đối lớn nhất : {max_abs:.3e}   (fp32 epsilon ~1.2e-7)")
print(f"  số lần top-k lệch (k=5/50/200/500): {rank_mismatch} / {n*4}")
print(f"\n  rank_bm25 {t_old/n*1000:8.1f} ms/câu")
print(f"  CSR       {t_new/n*1000:8.1f} ms/câu   -> nhanh hơn {t_old/t_new:.0f}×")
