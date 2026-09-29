"""Encode lại toàn bộ corpus bằng bi-encoder đã fine-tune, ghi ra ma trận dense.

Bắt buộc phải làm sau khi fine-tune retriever: vector của 184.368 chunk trong index cũ
được sinh bởi model CŨ, không so sánh được với vector query của model MỚI. Đây cũng là
lý do fine-tune retriever đắt hơn fine-tune reranker rất nhiều.

Thứ tự chunk giữ NGUYÊN như bge_dense.ids.json để mọi thứ downstream dùng lại được.
"""
import argparse, gzip, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def mean_pool(out, mask, mode="mean"):
    """e5 dùng mean pooling, BGE-M3 dùng [CLS], Qwen3/harrier dùng last-token. Sai
    kiểu pooling là hỏng embedding hoàn toàn — vector sinh ra không so được với query
    vector của chính model đó.

    last_token lấy theo attention_mask (không phải h[:, -1] cứng): padding bên phải
    (mặc định của tokenizer) khiến vị trí cuối cùng là token PAD, không phải token
    thật cuối câu."""
    h = out.last_hidden_state
    if mode == "cls":
        return h[:, 0]
    if mode == "last_token":
        idx = mask.sum(1) - 1
        return h[torch.arange(h.size(0), device=h.device), idx]
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


def main(a):
    with open(a.ids_file, encoding="utf-8") as f:
        meta = json.load(f)
    chunk_ids = meta["chunk_ids"]
    with gzip.open(a.texts, "rt", encoding="utf-8") as f:
        texts = json.load(f)
    idx = list(range(a.shard, len(chunk_ids), a.num_shards))
    print(f"shard {a.shard}/{a.num_shards}: {len(idx)}/{len(chunk_ids)} chunk")

    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        model = AutoModel.from_pretrained(a.model, add_pooling_layer=False)
    except TypeError:
        model = AutoModel.from_pretrained(a.model)
    model.half().cuda().eval()
    if a.reserve_gb > 0:
        import sys as _s, os as _o
        _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
        from eval.gpu_reserve import reserve_vram
        reserve_vram(a.reserve_gb, 0)

    # xếp theo độ dài để giảm đệm — nhanh hơn ~30% mà không đổi kết quả
    order = sorted(idx, key=lambda i: len(texts.get(chunk_ids[i], "")))
    out = np.zeros((len(idx), model.config.hidden_size), dtype=np.float16)
    pos = {g: k for k, g in enumerate(idx)}
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, len(order), a.batch_size):
            sl = order[s:s + a.batch_size]
            batch = [a.prefix + texts.get(chunk_ids[i], "") for i in sl]
            enc = tok(batch, padding=True, truncation=True, max_length=a.max_length,
                      return_tensors="pt")
            enc = {k: v.cuda() for k, v in enc.items()}
            v = F.normalize(mean_pool(model(**enc), enc["attention_mask"], a.pooling), dim=-1)
            for j, i in enumerate(sl):
                out[pos[i]] = v[j].cpu().numpy()
            if s % (a.batch_size * 100) == 0:
                done = s + len(sl)
                print(f"  {done}/{len(order)} · {(time.time()-t0)/max(done,1)*1000:.1f} ms/chunk",
                      flush=True)
    os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)
    np.save(f"{a.out_prefix}.shard{a.shard}of{a.num_shards}.npy", out)
    json.dump(idx, open(f"{a.out_prefix}.shard{a.shard}of{a.num_shards}.idx.json", "w"))
    print(f"✅ {a.out_prefix}.shard{a.shard}of{a.num_shards}.npy · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out_prefix", required=True)
    p.add_argument("--ids_file", default="retrieval/db_fast/bge_dense.ids.json")
    p.add_argument("--texts", default="retrieval/db_fast/texts.json.gz")
    p.add_argument("--prefix", default="passage: ")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--pooling", choices=["mean","cls","last_token"], default="mean")
    p.add_argument("--reserve_gb", type=float, default=0.0)
    main(p.parse_args())
