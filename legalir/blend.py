"""Bước 3/4 — LSE top-10 -> chấm đề mục bằng Qwen3 -> pha z -> cắt 5.

  a. Gộp LSE T=2 (legalir.aggregate) -> top-10 văn bản.
  b. Chấm (câu hỏi, đề mục văn bản) bằng Qwen3-Reranker-0.6B (legalir.rerank_qwen3).
  c. Điểm cuối = z(LSE) + α·z(đề mục), α = 0,15 cố định (không dò trên dev).
     Chuẩn hoá z theo TỪNG CÂU vì hai thang điểm khác nhau; α nhỏ nên đề mục chỉ đảo
     được các cặp sát nút — đúng vai trò tie-break đã đo (public 0,9519 -> 0,9554).
  d. Cắt 5, phá hoà tất định theo (-điểm, doc_id).

Ba cổng chặn cuối (raise, KHÔNG assert — `python -O` xoá sạch assert):
tập key trùng file câu hỏi · mỗi câu 1..5 văn bản · không trùng document_id.

    python -m legalir.blend --pool out/pool --scores out/pool/scores_ce.jsonl.gz \
        --questions private-official.json --out out/submission.json
"""
import argparse
import gzip
import json
import os

import numpy as np

from legalir.aggregate import rank_documents
from legalir.data import load_scores
from legalir.headings import load_headings

ALPHA = 0.15


def z(a):
    a = np.asarray(a, dtype=np.float64)
    s = a.std()
    return np.zeros_like(a) if s < 1e-12 else (a - a.mean()) / s


def channel_order(pool_dir):
    """{qid: [doc_id]} duyệt xen kẽ hạng của 4 kênh — chỉ dùng bù slot cho ca suy biến."""
    rows = {}
    for ch in ["bm25", "bge", "e5", "bgesparse"]:
        with gzip.open(os.path.join(pool_dir, f"chan_{ch}.jsonl.gz"), "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.setdefault(r["qid"], []).append(
                    [str(x[1]) for x in sorted(r["rows"], key=lambda x: -x[2])])
    out = {}
    for qid, lists in rows.items():
        seen, order = set(), []
        for i in range(max((len(l) for l in lists), default=0)):
            for l in lists:
                if i < len(l) and l[i] not in seen:
                    seen.add(l[i])
                    order.append(l[i])
        out[qid] = order
    return out


def blend_and_cut(top, lse, heading, alpha=ALPHA, top_k=5):
    out = {}
    for qid, docs in top.items():
        sq = heading.get(qid, {})
        floor = min(sq.values(), default=0.0) - 1.0      # văn bản không có đề mục
        total = z([lse[qid][d] for d in docs]) + alpha * z([sq.get(d, floor) for d in docs])
        out[qid] = [d for _, d in sorted(zip(-total, docs), key=lambda t: (t[0], t[1]))][:top_k]
    return out


def check(answers, qdata):
    if set(answers) != set(qdata):
        raise SystemExit(f"CHỐT HỎNG: tập key lệch file câu hỏi — thiếu "
                         f"{len(set(qdata) - set(answers))}, thừa {len(set(answers) - set(qdata))}")
    bad = [q for q, v in answers.items() if not 1 <= len(v) <= 5]
    if bad:
        raise SystemExit(f"CHỐT HỎNG: {len(bad)} câu có số văn bản ngoài [1,5] — scorer cho "
                         f"0 điểm câu đó. Ví dụ: {bad[:5]}")
    dup = [q for q, v in answers.items() if len(set(v)) != len(v)]
    if dup:
        raise SystemExit(f"CHỐT HỎNG: {len(dup)} câu trùng document_id. Ví dụ: {dup[:5]}")


def main():
    ap = argparse.ArgumentParser(prog="legalir.blend")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--headings", default="index/headings_v3_strict.jsonl")
    ap.add_argument("--heading_model", default="models/Qwen3-Reranker-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--top_x", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--max_length", type=int, default=1024)
    a = ap.parse_args()

    qdata = json.load(open(a.questions, encoding="utf-8"))
    heads = load_headings(a.headings)
    print(f"{len(qdata)} câu hỏi · {len(heads)} đề mục")

    top, lse = rank_documents(a.pool, load_scores(a.scores), top_x=a.top_x)

    pairs, owner = [], []
    for qid, docs in top.items():
        for d in docs:
            if d in heads:
                pairs.append((qdata[qid]["question"], heads[d]))
                owner.append((qid, d))
    print(f"{len(pairs)} cặp (câu hỏi, đề mục) cần chấm", flush=True)

    from legalir.rerank_qwen3 import HeadingReranker
    values = HeadingReranker(a.heading_model, a.max_length).score(pairs, a.batch_size)
    heading = {}
    for (qid, d), v in zip(owner, values):
        heading.setdefault(qid, {})[d] = float(v)

    answers = blend_and_cut(top, lse, heading, a.alpha)

    # Ca suy biến (0/1.000 câu public): pool rỗng hoặc gom được <5 văn bản. Recall + trần
    # 5 văn bản ⇒ slot trống chỉ có thể mất điểm, nên bù theo thứ hạng kênh truy hồi.
    short = [q for q in qdata if len(answers.get(q, [])) < 5]
    if short:
        fill = channel_order(a.pool)
        for q in short:
            picked = answers.get(q, [])
            for d in fill.get(q, ()):
                if len(picked) >= 5:
                    break
                if d not in picked:
                    picked.append(d)
            answers[q] = picked
        print(f"bù slot cho {len(short)} câu: {short[:10]}")

    answers = {q: answers[q] for q in qdata if q in answers}
    check(answers, qdata)
    print(f"=== CHỐT OK === {len(answers)} câu · mỗi câu 1..5 văn bản · không trùng")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({q: {"answer": v} for q, v in answers.items()},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"✅ {a.out}")


if __name__ == "__main__":
    main()
