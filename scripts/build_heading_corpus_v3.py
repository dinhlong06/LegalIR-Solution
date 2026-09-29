#!/usr/bin/env python
"""Dựng lại đề mục v3-strict từ corpus — kiểm tra tái lập: file ra phải trùng khít từng
byte với index/headings_v3_strict.jsonl đang đóng gói.

Mỗi văn bản -> chuỗi các đề mục cấu trúc, nối bằng " . ":
  - mốc đề mục: "Điều N." / "Chương IV ..." / "Mục 2." / "Phần I ..." — Chương/Mục/Phần
    phải có dấu . : hoặc chữ VIẾT HOA theo sau (loại câu trích dẫn chéo kiểu
    "quy định tại Mục 2, 3 Chương này");
  - mỗi đề mục lấy tối đa 14 từ; với "Điều" cắt trước khoản đầu tiên ("1. " / "a) ");
  - bỏ trùng (không phân biệt hoa thường), bỏ đề mục <= 5 ký tự, tối đa 200 đề mục;
  - văn bản không có mốc nào (công văn, bảng biểu): lấy 120 từ đầu.

Đầu ra: JSONL {"doc_id", "summary"} theo thứ tự tên file.

    python scripts/build_heading_corpus_v3.py --corpus_dir <selected-contexts> \
        --out /tmp/headings_v3_strict.jsonl
    diff /tmp/headings_v3_strict.jsonl index/headings_v3_strict.jsonl   # phải trùng khít
"""
import argparse
import glob
import json
import os
import re

MARK = re.compile(
    r"(Điều\s+\d+[a-zA-Z]?\s*[.:]"
    r"|Chương\s+[IVXLC]+\s*[.:]|Chương\s+[IVXLC]+\b(?=\s+"
    r"[A-ZĐÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ])"
    r"|Mục\s+\d+\s*[.:]|Mục\s+\d+\b(?=\s+"
    r"[A-ZĐÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]{2,})"
    r"|Phần\s+[IVXLC\d]+\s*[.:]|Phần\s+[IVXLC\d]+\b(?=\s+"
    r"[A-ZĐÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]{2,}))")
CLAUSE = re.compile(r"\s(?:\d{1,2}\.\s|[a-zđ]\)\s)")


def headings(text, words_after=14, max_heads=200):
    out, seen = [], set()
    marks = list(MARK.finditer(text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        seg = text[m.start():end]
        if seg.lstrip().startswith("Điều"):
            mm = MARK.match(seg)
            clause = CLAUSE.search(seg, mm.end() if mm else 0)
            if clause:
                seg = seg[:clause.start()]
        h = " ".join(seg.split()[:words_after]).strip(" .;,")
        if len(h) <= 5 or h.lower() in seen:
            continue
        seen.add(h.lower())
        out.append(h)
        if len(out) >= max_heads:
            break
    return out


def summary(passage, fallback_words=120):
    text = " ".join(passage.split())
    hs = headings(text)
    return " . ".join(hs) if hs else " ".join(text.split()[:fallback_words])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.corpus_dir, "*.json")))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fo:
        for path in files:
            doc_id = os.path.basename(path)[len("context_"):-len(".json")]
            passage = json.load(open(path, encoding="utf-8")).get("passage", "") or ""
            fo.write(json.dumps({"doc_id": doc_id, "summary": summary(passage)},
                                ensure_ascii=False) + "\n")
    print(f"✅ {len(files)} văn bản -> {a.out}")


if __name__ == "__main__":
    main()
