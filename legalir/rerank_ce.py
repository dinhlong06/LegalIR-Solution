"""Bước 2/4 — cross-encoder chấm từng cặp (câu hỏi, chunk) trong pool.

Khác bi-encoder ở chỗ câu hỏi và chunk vào CÙNG một lượt forward, nên model đọc được quan
hệ giữa chúng; đắt hơn nhiều nên chỉ chạy trên ~95 chunk/câu mà tầng truy hồi đã lọc.

`normalize=True` đưa điểm về [0,1] (sigmoid) — tầng gộp sau đó quy ngược về logit.
Điểm được ghi ra file nên đổi cách gộp/cắt không phải chấm lại GPU.

    python -m legalir.rerank_ce --pool out/pool --model models/ce_fam_ep1 \
        --out out/pool/scores_ce.jsonl.gz
"""
import argparse
import gzip
import json
import os
import time

from legalir.data import CE_CHANNELS, build_pool, load_rows, load_texts, parse_channels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", nargs="+", default=CE_CHANNELS)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    a = ap.parse_args()

    rows, texts = load_rows(a.pool), load_texts(a.pool)
    depths = parse_channels(a.channels)
    rows = [r for i, r in enumerate(rows) if i % a.num_shards == a.shard]
    print(f"shard {a.shard}/{a.num_shards}: {len(rows)} câu · độ sâu {depths}")

    pairs, spans = [], []
    for row in rows:
        cids = [c for c in build_pool(row, depths) if c in texts]
        start = len(pairs)
        pairs.extend([[row["question"], texts[c]] for c in cids])
        spans.append((row["qid"], start, len(pairs), cids))
    print(f"{len(pairs)} cặp · {len(pairs)/max(len(rows),1):.0f} chunk/câu", flush=True)

    from FlagEmbedding import FlagReranker
    model = FlagReranker(a.model, use_fp16=True)
    t0 = time.time()
    scores = model.compute_score(pairs, batch_size=a.batch_size,
                                 max_length=a.max_length, normalize=True)
    dt = time.time() - t0
    print(f"chấm {len(pairs)} cặp trong {dt:.0f}s ({len(pairs)/dt:.0f} cặp/s)")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with gzip.open(a.out, "wt", encoding="utf-8") as f:
        for qid, start, end, cids in spans:
            f.write(json.dumps({"qid": qid,
                                "scores": dict(zip(cids, [float(s) for s in scores[start:end]]))},
                               ensure_ascii=False) + "\n")
    json.dump({"model": a.model, "channels": a.channels, "n_pairs": len(pairs),
               "seconds": dt, "max_length": a.max_length, "batch_size": a.batch_size},
              open(a.out.replace(".jsonl.gz", ".meta.json"), "w"), indent=2)
    print(f"✅ {a.out}")


if __name__ == "__main__":
    main()
