"""Bước 4/4 — cổng chặn định dạng + đóng gói nộp.

Ba ràng buộc chết người của scorer BTC:
  - tối đa 5 document_id mỗi câu; trả 6 -> Recall = Precision = 0 cho câu đó;
  - tập key phải TRÙNG KHÍT tập câu hỏi; thiếu/thừa -> scorer raise -> 0 điểm toàn bài;
  - file nộp là .zip chứa ĐÚNG MỘT tệp tên `submission.json`.

    python -m legalir.validate validate --submission out/submission.json \
        --questions private-official.json --doc_ids index/doc_ids.json
    python -m legalir.validate package --submission out/submission.json --out out/submission.zip
"""
import argparse
import json
import sys
import zipfile


def _load(path):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if names != ["submission.json"]:
                raise SystemExit(f"⛔ zip phải chứa đúng 1 tệp `submission.json`, đang có {names}")
            raw = z.read("submission.json")
    else:
        raw = open(path, "rb").read()
    return json.loads(raw.decode("utf-8"))          # lỗi UTF-8 sẽ raise ngay ở đây


def validate(a):
    sub = _load(a.submission)
    questions = json.load(open(a.questions, encoding="utf-8"))
    corpus = set(json.load(open(a.doc_ids, encoding="utf-8"))) if a.doc_ids else None

    checks = [
        ("là JSON object", isinstance(sub, dict), type(sub).__name__),
        ("mỗi question_id đúng 1 lần", set(sub) == set(questions),
         f"thiếu {len(set(questions) - set(sub))} · thừa {len(set(sub) - set(questions))}"),
        ("question_id là chuỗi", all(isinstance(k, str) for k in sub), ""),
        ("mọi câu có trường `answer`", all("answer" in v for v in sub.values()), ""),
        ("`answer` là mảng chuỗi",
         all(isinstance(v.get("answer"), list)
             and all(isinstance(d, str) for d in v["answer"]) for v in sub.values()), ""),
        ("không câu nào quá 5 văn bản",
         all(len(v.get("answer", [])) <= 5 for v in sub.values()), ""),
        ("không câu nào rỗng", all(v.get("answer") for v in sub.values()), ""),
        ("không trùng document_id trong một câu",
         all(len(set(v["answer"])) == len(v["answer"]) for v in sub.values()), ""),
    ]
    if corpus is not None:
        unknown = {d for v in sub.values() for d in v.get("answer", [])} - corpus
        checks.append(("mọi document_id có trong corpus", not unknown,
                       f"{len(unknown)} id lạ: {sorted(unknown)[:5]}"))

    width = max(len(c[0]) for c in checks)
    bad = 0
    for name, ok, detail in checks:
        bad += not ok
        print(f"  {'✅' if ok else '❌'}  {name:<{width}}  {detail if not ok else ''}")
    n = len(sub)
    print(f"\n{n} câu · {sum(len(v['answer']) for v in sub.values())/max(n,1):.2f} văn bản/câu")
    print("HỢP LỆ" if not bad else f"CÓ {bad} LỖI — KHÔNG NỘP")
    return 1 if bad else 0


def package(a):
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(a.submission, arcname="submission.json")
    with zipfile.ZipFile(a.out) as z:
        if z.namelist() != ["submission.json"]:
            raise SystemExit(f"⛔ zip sai nội dung: {z.namelist()}")
    print(f"✅ {a.out}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "package"])
    ap.add_argument("--submission", required=True)
    ap.add_argument("--questions")
    ap.add_argument("--doc_ids", default="")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "validate":
        if not a.questions:
            raise SystemExit("validate cần --questions")
        return validate(a)
    if not a.out:
        raise SystemExit("package cần --out")
    return package(a)


if __name__ == "__main__":
    sys.exit(main())
