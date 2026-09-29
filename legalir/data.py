"""Nạp dữ liệu dùng chung: câu hỏi, pool, điểm CE, vân tay corpus.

Gộp ba module trước đây (io.py · pool.py · data/corpus_manifest.py) — chỉ nạp/kiểm,
không tính điểm.

Độ sâu pool ("bm25:50 bge:30 e5:30 bgesparse:30") cắt từng kênh trước khi hợp. Nhờ vậy
pool có thể được dựng ở độ sâu LỚN HƠN rồi mới cắt về độ sâu production — chấm một lần,
dùng lại cho nhiều cấu hình.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict


# ═══════════════ tiện ích đọc/ghi ═══════════════

def load_json(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path, indent=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def read_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_questions(path):
    """{qid: {"question": ...}} -> (qids đã sort, câu hỏi tương ứng)."""
    data = load_json(path)
    qids = sorted(data)
    return qids, [data[q]["question"] for q in qids]


def load_doc_ids(path):
    """Danh sách document_id hợp lệ của corpus (index/doc_ids.json)."""
    return set(json.load(open(path, encoding="utf-8")))


# ═══════════════ đọc pool do `legalir.retrieve` ghi ═══════════════

CE_CHANNELS = ["bm25:50", "bge:30", "e5:30", "bgesparse:30"]


def parse_channels(specs):
    out = {}
    for s in specs:
        name, _, depth = s.partition(":")
        out[name] = int(depth) if depth else 10 ** 9
    return out


def load_rows(pool_dir):
    """[{qid, question, channels:{kênh: rows}}], sắp theo (độ dài qid, qid) — thứ tự này
    quyết định thứ tự cặp đưa vào cross-encoder, giữ nguyên để tái lập từng bit."""
    merged, questions = {}, {}
    names = []
    for path in sorted(os.listdir(pool_dir)):
        if not (path.startswith("chan_") and path.endswith(".jsonl.gz")):
            continue
        name = path[len("chan_"):-len(".jsonl.gz")]
        names.append(name)
        with gzip.open(os.path.join(pool_dir, path), "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                questions[r["qid"]] = r["question"]
                merged.setdefault(r["qid"], {})[name] = r["rows"]
    print(f"nạp {len(names)} kênh: {names}")
    return [{"qid": q, "question": questions[q], "channels": merged[q]}
            for q in sorted(merged, key=lambda x: (len(x), x))]


def load_texts(pool_dir):
    with gzip.open(os.path.join(pool_dir, "texts_fast.json.gz"), "rt", encoding="utf-8") as f:
        return json.load(f)


def build_pool(row, depths):
    """{chunk_id: doc_id} theo thứ tự chèn (kênh theo thứ tự `depths`)."""
    pool = {}
    for ch, depth in depths.items():
        for cid, aid, _score in row["channels"].get(ch, [])[:depth]:
            pool.setdefault(cid, aid)
    return pool


def pool_chunks(pool_dir, channels):
    """{qid: set(chunk_id)} ở đúng độ sâu cho trước — dùng khi chấm lại từ điểm đã cache."""
    per_q = defaultdict(set)
    for spec in channels:
        chan, n = spec.split(":")
        n = int(n)
        path = os.path.join(pool_dir, f"chan_{chan}.jsonl.gz")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows = sorted(r["rows"], key=lambda x: -x[2])[:n]
                per_q[r["qid"]].update(x[0] for x in rows)
    return per_q


def load_scores(path):
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.setdefault(r["qid"], {}).update(r["scores"])
    print(f"nạp {len(out)} câu · {sum(len(v) for v in out.values())} cặp đã chấm")
    return out


# ═══════════════ vân tay corpus (cổng: corpus phải trùng corpus đã dựng index) ═══════════════

def fingerprint(corpus_dir):
    out = {}
    for p in sorted(glob.glob(os.path.join(corpus_dir, "context_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        out[str(d["id"])] = hashlib.sha1((d.get("passage") or "").encode("utf-8")).hexdigest()
    return out


def check_corpus(corpus_dir, manifest="index/corpus_manifest.json"):
    """Trả 0 nếu corpus khớp index đóng gói, 1 nếu lệch (phải dựng lại index)."""
    fp = fingerprint(corpus_dir)
    if not fp:
        print(f"⛔ không thấy context_*.json trong {corpus_dir}")
        return 1
    ref = json.load(open(manifest, encoding="utf-8"))
    new_ids, gone = set(fp) - set(ref), set(ref) - set(fp)
    changed = [k for k in set(fp) & set(ref) if fp[k] != ref[k]]
    print(f"corpus {len(fp)} văn bản · manifest {len(ref)} · mới {len(new_ids)} · "
          f"mất {len(gone)} · đổi nội dung {len(changed)}")
    if new_ids or changed:
        print("⛔ corpus KHÁC corpus đã dựng index — phải dựng lại index (README §6). Dừng.")
        return 1
    if gone:
        print("⚠️  corpus thiếu văn bản so với index — vẫn chạy được; văn bản thiếu có thể "
              "xuất hiện trong kết quả.")
    print("✅ corpus khớp index đóng gói")
    return 0


def main():
    ap = argparse.ArgumentParser(prog="legalir.data")
    ap.add_argument("cmd", choices=["manifest"])
    ap.add_argument("--corpus_dir", required=True)
    ap.add_argument("--manifest", default="index/corpus_manifest.json")
    ap.add_argument("--doc_ids", default="index/doc_ids.json")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    fp = fingerprint(a.corpus_dir)
    if not fp:
        print(f"⛔ không thấy context_*.json trong {a.corpus_dir}")
        return 1
    if a.build:
        json.dump(fp, open(a.manifest, "w", encoding="utf-8"), indent=0, sort_keys=True)
        json.dump(sorted(fp), open(a.doc_ids, "w", encoding="utf-8"))
        print(f"✅ manifest {len(fp)} văn bản -> {a.manifest}, {a.doc_ids}")
        return 0
    return check_corpus(a.corpus_dir, a.manifest)


if __name__ == "__main__":
    sys.exit(main())
