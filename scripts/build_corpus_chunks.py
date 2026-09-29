#!/usr/bin/env python
"""Bước dựng index 1/6 — corpus BTC -> chunk + universe chunk.

    selected-contexts/context_*.json  ->  index/db_fast/texts.json.gz
                                          index/db_fast/chunk_ids.json

Đây là mắt xích GỐC: đổi cách chunk là đổi thứ tự chunk ⇒ mọi ma trận dense,
sparse, BM25 đều phải dựng lại. Logic tạo chunk chép NGUYÊN VĂN từ
`training_logs/original_scripts/index_build/build_db_corpus_dsc.py` (bỏ phần
Qdrant/encoder — các bước sau do `legalir.index.*` lo).

Quy ước ID (bất biến toàn hệ thống, KHÔNG được đổi):
    chunk_id = f"{aid}_{ordinal}_{chunk_n}"
      aid      = trường "id" của văn bản (fallback "unknown_<vị trí>")
      ordinal  = vị trí file theo tên đã sort, 1-based — KỂ CẢ file passage rỗng
                 vẫn chiếm một ordinal (nếu bỏ sót, toàn bộ ID lệch)
      chunk_n  = thứ tự chunk trong văn bản, 1-based

Ghi gzip TẤT ĐỊNH (mtime=0, không nhúng tên file) để nội dung tái lập được giữa
các lần chạy; so sánh với bản đóng gói nên so NỘI DUNG, không so hash file.

    python scripts/build_corpus_chunks.py --corpus_dir <selected-contexts> \
        --out_dir index/db_fast
"""
import argparse
import glob
import gzip
import json
import os
import re
import sys


def count_tokens(text):
    return len(text.split())


def get_last_overlap(text):
    section_pattern = r'[^.!?]+[.!?]'
    sections = re.findall(section_pattern, text.strip(), flags=re.DOTALL)
    if sections:
        return sections[-1].strip()
    paras = re.split(r'\n\s*\n', text.strip())
    if len(paras) > 0:
        return paras[-1].strip()
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    if len(sentences) > 0:
        return sentences[-1].strip()
    return text.strip()


def split_chunks(text, max_len=2000):
    section_pattern = r'[^.!?]+[.!?]'
    sections = re.findall(section_pattern, text, flags=re.DOTALL)
    if not sections:
        sections = [text.strip()]
    raw_chunks, current_chunk_content = [], ""
    for section in sections:
        section = section.strip()
        tokens_to_add = count_tokens(section) + (1 if current_chunk_content else 0)
        if count_tokens(current_chunk_content) + tokens_to_add <= max_len:
            current_chunk_content = current_chunk_content + "\n" + section if current_chunk_content else section
        else:
            if current_chunk_content:
                raw_chunks.append(current_chunk_content)
            if count_tokens(section) > max_len:
                paras = re.split(r'\n\s*\n', section)
                sub_chunk_content = ""
                for para in paras:
                    para = para.strip()
                    para_tokens_to_add = count_tokens(para) + (2 if sub_chunk_content else 0)
                    if count_tokens(sub_chunk_content) + para_tokens_to_add <= max_len:
                        sub_chunk_content = sub_chunk_content + "\n\n" + para if sub_chunk_content else para
                    else:
                        if sub_chunk_content:
                            raw_chunks.append(sub_chunk_content)
                        sub_chunk_content = para
                if sub_chunk_content:
                    raw_chunks.append(sub_chunk_content)
                current_chunk_content = ""
            else:
                current_chunk_content = section
    if current_chunk_content:
        raw_chunks.append(current_chunk_content)
    final_chunks = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            final_chunks.append(chunk)
        else:
            overlap = get_last_overlap(raw_chunks[i - 1])
            current_overlap = overlap
            while True:
                potential_new_chunk = (current_overlap + "\n\n" + chunk).strip()
                if count_tokens(potential_new_chunk) <= max_len:
                    final_chunks.append(potential_new_chunk)
                    break
                if len(current_overlap.split()) > 1:
                    current_overlap = ' '.join(current_overlap.split()[1:])
                elif len(current_overlap) > 0:
                    current_overlap = current_overlap[1:]
                else:
                    final_chunks.append(chunk)
                    break
    return final_chunks


def build(corpus_dir, out_dir, max_len=450):
    files = sorted(glob.glob(os.path.join(corpus_dir, "*.json")))
    if not files:
        raise SystemExit(f"⛔ không thấy *.json trong {corpus_dir}")
    chunk_ids, aids, texts = [], [], {}
    n_blank = 0
    for index, path in enumerate(files):
        d = json.load(open(path, encoding="utf-8"))
        aid = d.get("id", f"unknown_{index}")
        passage = str(d.get("passage", ""))
        if not passage.strip():                    # văn bản rỗng: không có chunk, VẪN chiếm ordinal
            n_blank += 1
            continue
        for j, chunk in enumerate(split_chunks(passage, max_len=max_len)):
            cid = f"{aid}_{index + 1}_chunk{j + 1}"
            chunk_ids.append(cid)
            aids.append(str(aid))
            texts[cid] = chunk.strip()
    os.makedirs(out_dir, exist_ok=True)
    with gzip.GzipFile(os.path.join(out_dir, "texts.json.gz"), "wb", mtime=0) as f:
        f.write(json.dumps(texts, ensure_ascii=False).encode("utf-8"))
    json.dump({"chunk_ids": chunk_ids, "aids": aids},
              open(os.path.join(out_dir, "chunk_ids.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"✅ {len(files)} văn bản ({n_blank} rỗng bị loại) -> {len(chunk_ids)} chunk -> {out_dir}")
    return len(chunk_ids)


def main():
    ap = argparse.ArgumentParser(prog="build_corpus_chunks")
    ap.add_argument("--corpus_dir", required=True, help="thư mục context_*.json của BTC")
    ap.add_argument("--out_dir", required=True, help="thường là index/db_fast")
    ap.add_argument("--max_len", type=int, default=450, help="token/chunk (KHÔNG đổi: đổi là đổi toàn bộ index)")
    a = ap.parse_args()
    build(a.corpus_dir, a.out_dir, a.max_len)


if __name__ == "__main__":
    sys.exit(main())
