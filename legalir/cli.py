"""`legalir <questions.json> [fam] [out_dir]` — suy luận end-to-end.

Mặc định `fam` = bài private tốt nhất 937899 `fam_twinall_nosalary` (0,9491).

Thay `run_private.sh`: năm bước, mỗi bước một module, chạy bằng ĐÚNG interpreter của môi
trường tương ứng — hai tầng cần hai bản `transformers` không tương thích:
    PY_RETR  (mặc định /opt/venv_retr/bin/python)  transformers 5.16.1 · ST 6.0.0
             -> bước 1 (retrieval) + bước 3 (Qwen3-Reranker chấm đề mục)
    PY_CE    (mặc định /opt/venv_ce/bin/python)    transformers 4.44.2 · FlagEmbedding 1.2.11
             -> bước 2 (cross-encoder) + bước 4 (luật sinh đôi R3) + bước 5 (kiểm định dạng)

Chạy lại được giữa chừng: bước đã có kết quả đầy đủ (pool, điểm cross-encoder) thì bỏ qua.
Biến môi trường thêm: CORPUS_DIR (thư mục context_*.json — nếu đặt thì kiểm corpus khớp
index), CUDA_VISIBLE_DEVICES (một GPU >= 11 GB; đỉnh ~5 GB).

Module này chỉ dùng stdlib + PyYAML (cả hai venv đều có) — chạy được ở bất kỳ venv nào.
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
import time

from legalir.config import load_config


def _n_pool(pool_dir):
    """Số câu đã có pool (đếm dòng chan_bm25) — None nếu chưa có."""
    path = os.path.join(pool_dir, "chan_bm25.jsonl.gz")
    if not os.path.isfile(path):
        return 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="legalir",
                                 description="LegalIR: questions.json -> submission.zip")
    ap.add_argument("questions", help="file câu hỏi (json, {qid: {question: ...}})")
    ap.add_argument("config", nargs="?", default="fam",
                    help="fam (cấu hình duy nhất) | đường dẫn .yaml")
    ap.add_argument("out_dir", nargs="?", help="mặc định output/<config>")
    a = ap.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)                                     # mọi đường dẫn (index/, models/) tính từ gói
    cfg = load_config(a.config, os.path.join(root, "configs"))
    out = a.out_dir or os.path.join("output", cfg.name)
    pool = os.path.join(out, "pool")
    scores = os.path.join(pool, "scores_ce.jsonl.gz")
    base = os.path.join(out, "base_lse_heading.json")   # 5 văn bản sau bước 3
    sub = os.path.join(out, "submission.json")
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    log = open(os.path.join(out, "logs", "run.log"), "a", encoding="utf-8")

    def say(msg):
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    def run(py, args, logfile):
        with open(os.path.join(out, "logs", logfile), "a", encoding="utf-8") as lf:
            return subprocess.run([py, "-u"] + args, stdout=lf, stderr=subprocess.STDOUT).returncode

    py_retr = os.environ.get("PY_RETR", "/opt/venv_retr/bin/python")
    py_ce = os.environ.get("PY_CE", "/opt/venv_ce/bin/python")
    env = dict(os.environ)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONPATH=root)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.update(env)

    with open(a.questions, encoding="utf-8") as f:
        nq = len(json.load(f))
    say(f"=== LegalIR · cấu hình {cfg.name} · {nq} câu · {a.questions} -> {out} ===")
    say(f"retrieval: {cfg.dense_model} + {cfg.e5_model} (depth {cfg.topk_bm25}/{cfg.topk_dense}) · "
        f"CE: {cfg.ce_model}")

    # [0] corpus phải trùng corpus đã dựng index -----------------------------------
    corpus_dir = os.environ.get("CORPUS_DIR", "")
    if corpus_dir:
        if run(py_ce, ["-m", "legalir.data", "manifest", "--corpus_dir", corpus_dir],
               "step0_corpus.log") != 0:
            say("⛔ corpus lệch index — dừng")
            return 1
    else:
        say("ℹ️  không đặt CORPUS_DIR — bỏ qua kiểm corpus")

    # [1] truy hồi 4 kênh ------------------------------------------------------------
    if _n_pool(pool) == nq and os.path.isfile(os.path.join(pool, "texts_fast.json.gz")):
        say(f"[1/5] pool đã có ({nq} câu) — bỏ qua")
    else:
        say(f"[1/5] truy hồi: BM25 + bge-m3 sparse + {cfg.dense_model} + {cfg.e5_model} "
            f"(depth {cfg.topk_bm25}/{cfg.topk_dense})")
        if run(py_retr, ["-m", "legalir.retrieve",
                         "--questions", a.questions, "--out_dir", pool,
                         "--index_dir", "index/db_fast", "--bge_model", "models/bge-m3",
                         "--dense_model", cfg.dense_model, "--dense_prefix", cfg.dense_prefix,
                         "--e5_model", cfg.e5_model, "--e5_prefix", cfg.e5_prefix,
                         "--top_k_bm25", str(cfg.topk_bm25),
                         "--top_k_dense", str(cfg.topk_dense)], "step1_retrieve.log") != 0 \
                or _n_pool(pool) != nq:
            say(f"⛔ bước 1 lỗi — xem {out}/logs/step1_retrieve.log")
            return 1

    # [2] cross-encoder chấm chunk ---------------------------------------------------
    if os.path.isfile(scores):
        say("[2/5] điểm cross-encoder đã có — bỏ qua")
    else:
        say(f"[2/5] cross-encoder {cfg.ce_model}")
        if run(py_ce, ["-m", "legalir.rerank_ce", "--pool", pool,
                       "--model", cfg.ce_model, "--out", scores],
               "step2_cross_encoder.log") != 0 or not os.path.isfile(scores):
            say(f"⛔ bước 2 lỗi — xem {out}/logs/step2_cross_encoder.log")
            return 1

    # [3] gộp LSE + đề mục Qwen3 -> 5 văn bản ----------------------------------------
    say("[3/5] gộp LSE T=2 + đề mục Qwen3-Reranker-0.6B (α=0,15) + bù slot câu < 5 văn bản")
    if run(py_retr, ["-m", "legalir.blend", "--pool", pool, "--scores", scores,
                     "--questions", a.questions, "--headings", "index/headings_v3_strict.jsonl",
                     "--heading_model", "models/Qwen3-Reranker-0.6B", "--out", base],
           "step3_blend.log") != 0:
        say(f"⛔ bước 3 lỗi — xem {out}/logs/step3_blend.log")
        return 1

    # [4] luật hậu kỳ R3 câu sinh đôi (0 tham số neural) -----------------------------
    say("[4/5] luật hậu kỳ R3 câu sinh đôi (train.json, cos ≥ 0,70, chèn hết gold hàng xóm)")
    if run(py_ce, ["-m", "legalir.rules", "--base", base, "--questions", a.questions,
                   "--train", "index/train.json", "--out", sub,
                   "--log", os.path.join(out, "logs", "r3_changes.json")], "step4_rules.log") != 0:
        say(f"⛔ bước 4 lỗi — xem {out}/logs/step4_rules.log")
        return 1

    # [5] cổng định dạng + đóng gói --------------------------------------------------
    say("[5/5] kiểm định dạng + đóng gói")
    if run(py_ce, ["-m", "legalir.validate", "validate", "--submission", sub,
                   "--questions", a.questions, "--doc_ids", "index/doc_ids.json"],
           "step5_validate.log") != 0:
        say("⛔ submission KHÔNG hợp lệ — không nộp")
        return 1
    if run(py_ce, ["-m", "legalir.validate", "package", "--submission", sub,
                   "--out", os.path.join(out, "submission.zip")], "step5_validate.log") != 0:
        return 1
    if run(py_ce, ["-m", "legalir.validate", "validate",
                   "--submission", os.path.join(out, "submission.zip"),
                   "--questions", a.questions, "--doc_ids", "index/doc_ids.json"],
           "step5_validate.log") != 0:
        say("⛔ zip không hợp lệ")
        return 1
    say(f"✅ SẴN SÀNG NỘP: {os.path.join(out, 'submission.zip')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
