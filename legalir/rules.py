"""Bước 4/5 — luật hậu kỳ R3 "câu sinh đôi", 0 tham số neural, áp lên 5 văn bản của bước 3.

  R3  Câu test có hàng xóm trong train.json với char-TFIDF cos ≥ 0,70 (2..5-gram, fit THUẦN
      trên train), qua lọc lệch thực thể (Jaccard tập từ có df ≤ 100 và dài > 2 ký tự ≥ 0,5)
      ⇒ chèn HẾT gold của hàng xóm chưa nằm trong top-5, lấp từ slot cuối ngược lên
      (slot 5, 4, 3, …). KHÔNG ràng buộc ứng viên phải có trong pool.

Bài private 937899 `fam_twinall_nosalary` = **0,9491** (result.md §91–§92): trên private 2.080
câu, 405 câu có hàng xóm ≥ 0,70, đổi 68 câu (6 câu chèn 2 gold).

Ngưỡng CỐ ĐỊNH, không dò trên tập nộp. Chỉ dùng train.json của BTC để fit vectorizer;
câu hỏi test chỉ được transform lúc suy luận.

    python -m legalir.rules --base out/base.json --questions private-official.json \
        --train index/train.json --out out/submission.json [--log out/r3_changes.json]
"""
import argparse
import json
import re
import unicodedata
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

SIM, DF_MAX, JAC = 0.70, 100, 0.5


def deaccent(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").replace("Đ", "D").replace("đ", "d")


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", deaccent(s).lower()).split())


def neighbors(testq, train_q, sim=SIM, df_max=DF_MAX, jac_min=JAC):
    """{qid test: (qid train sinh đôi, cos)} cho các câu qua cổng cos + lọc thực thể."""
    tk, pk = sorted(train_q), sorted(testq)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    vec.fit([norm(train_q[q]) for q in tk])
    Xt = vec.transform([norm(train_q[q]) for q in tk]).astype(np.float32)
    Xp = vec.transform([norm(testq[q]) for q in pk]).astype(np.float32)
    S = np.asarray((Xp @ Xt.T).todense())
    nn, nv = S.argmax(1), S.max(1)
    df = Counter()
    ttok = {q: set(norm(train_q[q]).split()) for q in tk}
    for q in tk:
        df.update(ttok[q])
    rare = lambda ts: {t for t in ts if df[t] <= df_max and len(t) > 2}
    jac = lambda x, y: 1.0 if not (x | y) else len(x & y) / len(x | y)
    out = {}
    for j, q in enumerate(pk):
        if nv[j] < sim:
            continue
        tw = tk[nn[j]]
        if jac(rare(set(norm(testq[q]).split())), rare(ttok[tw])) < jac_min:
            continue
        out[q] = (tw, float(nv[j]))
    return out, int((nv >= sim).sum())


def twin_rule(sub, testq, train_q, train_gold):
    """Chèn hết gold của hàng xóm chưa có trong top-5, lấp slot 5, 4, 3, … Trả (bài mới, nhật ký, số câu ≥ SIM)."""
    nb, n_nb = neighbors(testq, train_q)
    new, changed = {q: list(sub[q]) for q in sorted(testq)}, []
    for q, (tw, cos) in nb.items():
        cand, seen = [], set(sub[q])
        for d in train_gold[tw]:
            if d not in seen:
                cand.append(d)
                seen.add(d)
        if not cand:
            continue
        cur = list(sub[q])
        slots = [4, 3, 2, 1, 0][:len(cand)]
        out = [cur[i] for i in slots]
        for i, d in zip(slots, cand):
            cur[i] = d
        if len(set(cur)) != 5:
            raise SystemExit(f"trùng document ở {q}")
        new[q] = cur
        changed.append(dict(qid=q, sim=round(cos, 3), twin=tw, ins=cand[:len(slots)], out=out,
                            n_gold_twin=len(train_gold[tw])))
    return new, changed, n_nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="5 văn bản/câu của bước 3")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--train", default="index/train.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", help="nhật ký từng câu bị đổi (json)")
    a = ap.parse_args()

    qj = json.load(open(a.questions, encoding="utf-8"))
    testq = {q: (v["question"] if isinstance(v, dict) else v) for q, v in qj.items()}
    raw = json.load(open(a.base, encoding="utf-8"))
    sub = {q: [str(x) for x in (v["answer"] if isinstance(v, dict) else v)] for q, v in raw.items()}
    if set(sub) != set(testq):
        raise SystemExit(f"bài nền {len(sub)} câu ≠ file câu hỏi {len(testq)} câu")
    if any(len(v) != 5 or len(set(v)) != 5 for v in sub.values()):
        raise SystemExit("bài nền không phải 5 văn bản riêng biệt/câu — bước 3 phải bù slot trước")
    train = json.load(open(a.train, encoding="utf-8"))
    tq = {q: v["question"] for q, v in train.items()}
    tg = {q: [str(x) for x in v["answer"]] for q, v in train.items()}

    new, changed, n_nb = twin_rule(sub, testq, tq, tg)
    n_ins = Counter(len(c["ins"]) for c in changed)
    print(f"R3 câu sinh đôi: {n_nb} câu có hàng xóm cos ≥ {SIM} · đổi {len(changed)} câu · "
          f"phân bố số văn bản chèn {dict(sorted(n_ins.items()))}")

    bad = [q for q, v in new.items() if len(v) != 5 or len(set(v)) != 5]
    if set(new) != set(testq) or bad:
        raise SystemExit(f"CHỐT luật hậu kỳ trượt: {len(bad)} câu sai số văn bản")
    json.dump({q: {"answer": new[q]} for q in sorted(new)}, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    if a.log:
        json.dump(changed, open(a.log, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✅ {a.out}")


if __name__ == "__main__":
    main()
