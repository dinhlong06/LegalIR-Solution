# Vendored từ eval/citation_graph.py (repo dev). Giữ nguyên toàn bộ logic build/regex —
# chỉ thay đổi: (1) CTX/CACHE thành tham số (mặc định theo layout gói thay vì
# data/... và eval/... của repo dev); (2) main() phân tích dev-ceiling (cần file dev
# của repo) được thay bằng CLI dựng graph thuần túy. Bản gốc đầy đủ vẫn nằm ở
# training_logs/original_scripts/ (không có ở đây) và eval/citation_graph.py (repo dev).
"""Do TRẦN của cơ chế 'mở rộng theo trích dẫn' cho các câu multi-gold.

Rule Graph cũ (eval/build_rule_graph.py) chỉ đọc SLUG của link nên ra 494 nút — quá
nghèo. Nguồn thật sự giàu là THÂN VĂN BẢN: phần "Căn cứ …", "sửa đổi, bổ sung một số
điều của Nghị định số 35/2016/NĐ-CP", "hướng dẫn Luật Thú y" nằm trong `passage`.

Ở đây:
  1. gán SỐ HIỆU cho từng văn bản, suy từ slug của link (một chiều, không mơ hồ);
  2. quét `passage` lấy mọi số hiệu được trích dẫn -> cạnh A -> B;
  3. đo trần: trong các gold doc bị TRƯỢT, bao nhiêu cái có cạnh nối với một doc ta ĐÃ
     trả về? Đó là mức trần tuyệt đối của mọi luật mở rộng theo trích dẫn.

Chỉ dùng dữ liệu BTC (selected-contexts). Không nguồn ngoài.
"""
import argparse
import glob
import json
import os
import re
import unicodedata
from collections import defaultdict

# Fallback khi gọi build() mà không truyền corpus_dir (giữ đường dẫn repo dev gốc
# để hành vi mặc định không đổi; driver trong gói luôn truyền explicit).
CTX = "data/LegalIR_Public_Test/selected-contexts/selected-contexts"


def deaccent(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").replace("Đ", "D").replace("đ", "d")


# Số hiệu trong THÂN văn bản: 35/2016/ND-CP · 12/2017/QH14 · 595/QD-BHXH · 56-HD/VPTW
CITE_FULL = re.compile(r"(?<![\d/])(\d{1,4})\s*/\s*(\d{4})\s*/\s*([A-ZD][A-Z\-]{1,12})(?![A-Z])")
CITE_SHORT = re.compile(r"(?<![\d/])(\d{1,4})\s*/\s*([A-ZD]{2,6}-[A-ZD]{2,8})(?![A-Z])")
# Số hiệu trong SLUG: nghi-dinh-35-2016-nd-cp-... / quyet-dinh-595-qd-bhxh-...
SLUG_FULL = re.compile(r"(?<![0-9a-z])(\d{1,4})-(\d{4})-([a-z]{2,5}(?:-[a-z]{2,8})?)(?![a-z0-9])")
SLUG_SHORT = re.compile(r"(?<![0-9a-z])(\d{1,4})-([a-z]{2,6}-[a-z]{2,8})(?![a-z0-9])")


def norm_full(num, year, code):
    return f"{int(num)}/{year}/{deaccent(code).upper().replace('-', '')}"


def norm_short(num, code):
    return f"{int(num)}/{deaccent(code).upper().replace('-', '')}"


CACHE = "index/citation_graph.json"


def build(verbose=True, cache=CACHE, corpus_dir=None):
    # Quét 8.532 passage (~40M từ) mất ~90s; kết quả tất định nên cache lại.
    if cache and os.path.exists(cache):
        j = json.load(open(cache, encoding="utf-8"))
        if verbose:
            print(f"đọc lại {cache}: {len(j['undirected'])} văn bản có cạnh")
        return (j["own"], {k: set(v) for k, v in j["edges"].items()},
                {k: set(v) for k, v in j["undirected"].items()})
    own, edges, und = _build(verbose, corpus_dir or CTX)
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        json.dump({"own": own,
                   "edges": {k: sorted(v) for k, v in edges.items()},
                   "undirected": {k: sorted(v) for k, v in und.items()}},
                  open(cache, "w", encoding="utf-8"))
    return own, edges, und


def _build(verbose=True, corpus_dir=None):
    corpus_dir = corpus_dir or CTX
    ids, slugs, passages = [], {}, {}
    for p in glob.glob(os.path.join(corpus_dir, "*.json")):
        d = json.load(open(p, encoding="utf-8"))
        did = str(d["id"])
        ids.append(did)
        slugs[did] = deaccent((d.get("link") or "").rsplit("/", 1)[-1]).lower()
        passages[did] = d.get("passage", "")

    # --- 1. số hiệu của chính văn bản, lấy CỤM ĐẦU TIÊN trong slug -----------
    # Slug bắt đầu bằng loại + số hiệu rồi mới tới tiêu đề, nên cụm khớp đầu tiên là
    # số hiệu của chính nó; các cụm sau (nếu có) là văn bản mà tiêu đề nhắc tới.
    own, cited_in_title = {}, defaultdict(set)
    for did in ids:
        s = slugs[did]
        hits = [(m.start(), norm_full(*m.groups())) for m in SLUG_FULL.finditer(s)]
        hits += [(m.start(), norm_short(*m.groups())) for m in SLUG_SHORT.finditer(s)]
        hits.sort()
        if hits:
            own[did] = hits[0][1]
            for _, c in hits[1:]:
                cited_in_title[did].add(c)

    by_code = defaultdict(list)
    for did, c in own.items():
        by_code[c].append(did)
    if verbose:
        print(f"{len(ids)} văn bản · gán được số hiệu cho {len(own)} "
              f"({100*len(own)/len(ids):.0f}%) · {len(by_code)} số hiệu khác nhau")

    # --- 2. trích dẫn trong thân văn bản ------------------------------------
    edges = defaultdict(set)
    for did in ids:
        # PHẢI bỏ dấu trước khi regex: "NĐ-CP" có chữ Đ, lớp ký tự [A-Z] không chứa nó,
        # mà NĐ-CP chính là loại văn bản được trích dẫn nhiều nhất.
        txt = deaccent(passages[did])
        found = set()
        for m in CITE_FULL.finditer(txt):
            found.add(norm_full(*m.groups()))
        for m in CITE_SHORT.finditer(txt):
            found.add(norm_short(*m.groups()))
        found |= cited_in_title[did]
        found.discard(own.get(did))
        for c in found:
            for tgt in by_code.get(c, []):
                if tgt != did:
                    edges[did].add(tgt)

    undirected = defaultdict(set)
    for a, bs in edges.items():
        for b in bs:
            undirected[a].add(b)
            undirected[b].add(a)
    if verbose:
        n_e = sum(len(v) for v in edges.values())
        print(f"{n_e} cạnh có hướng · {len(edges)} văn bản có ít nhất 1 cạnh đi ra · "
              f"{len(undirected)} văn bản có ít nhất 1 cạnh (vô hướng)")
    return own, edges, undirected


def main():
    ap = argparse.ArgumentParser(prog="legalir.train.citation_graph")
    ap.add_argument("--corpus_dir", required=True, help="thư mục context_*.json của BTC")
    ap.add_argument("--out", required=True, help="ghi {own, edges, undirected} (list đã sort)")
    a = ap.parse_args()
    own, edges, und = _build(verbose=True, corpus_dir=a.corpus_dir)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"own": own,
               "edges": {k: sorted(v) for k, v in edges.items()},
               "undirected": {k: sorted(v) for k, v in und.items()}},
              open(a.out, "w", encoding="utf-8"))
    print(f"✅ {a.out}")


if __name__ == "__main__":
    main()
