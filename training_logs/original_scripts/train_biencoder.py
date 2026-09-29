"""Fine-tune bi-encoder (retriever) bằng InfoNCE. Chỉ dữ liệu BTC.

Vì sao đáng làm dù đắt: đo trên dev, 34,2% phần đang trượt là gold doc KHÔNG LỌT
VÀO POOL — reranker giỏi tới đâu cũng không nhìn thấy chúng. Chỉ retriever tốt hơn
mới chạm tới, và nó nâng trần cho mọi tầng phía sau.

Loss: InfoNCE trên cosine, mỗi query cạnh tranh với
  - positive của chính nó
  - hard negative của chính nó (đã đào ở vòng 2, từ hạng 3-25 của reranker fine-tune)
  - positive của MỌI query khác trong batch (in-batch negative, gần như miễn phí)
Nên batch càng lớn càng nhiều negative — đó là lý do gom nhiều GPU.

e5 bắt buộc tiền tố 'query: ' / 'passage: '; sai tiền tố là hỏng embedding.
"""
import argparse, gzip, json, os, random, sys, time
from contextlib import nullcontext as _nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
if __package__:
    from .gold_negative_mask import load_gold_file, attach_gold, batch_gold_mask, mask_known_gold
else:
    from gold_negative_mask import load_gold_file, attach_gold, batch_gold_mask, mask_known_gold


def mean_pool(out, mask, mode="mean"):
    """e5 dùng mean pooling, BGE-M3 dùng [CLS]. Sai kiểu pooling là hỏng embedding
    hoàn toàn — vector sinh ra không so được với query vector của chính model đó."""
    h = out.last_hidden_state
    if mode == "cls":
        return h[:, 0]
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


class PairData(Dataset):
    def __init__(self, path, texts, n_neg, seed, max_pos=1, gold_file=None):
        with open(path, encoding="utf-8") as source:
            self.rows = [json.loads(l) for l in source]
        self.rows = [r for r in self.rows
                     if any(p in texts for p in r["pos"]) and any(n in texts for n in r["neg"])]
        self.texts, self.n_neg, self.rng = texts, n_neg, random.Random(seed)
        self.max_pos = max_pos
        self.gold = load_gold_file(gold_file, self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if self.max_pos > 1:
            return self._multi(r)
        pos = self.rng.choice([p for p in r["pos"] if p in self.texts])
        negs = [n for n in r["neg"] if n in self.texts]
        negs = (self.rng.sample(negs, self.n_neg) if len(negs) >= self.n_neg
                else negs + [self.rng.choice(negs) for _ in range(self.n_neg - len(negs))])
        sample = (r["query"], self.texts[pos], [self.texts[n] for n in negs])
        return attach_gold(sample, [pos] + negs, r, self.gold)

    def _multi(self, r):
        """--max_pos > 1 (13/09): lấy MỌI positive (mỗi văn bản gold một chunk), tối đa
        max_pos, đệm bằng bản lặp và đánh dấu ô đệm là không hợp lệ. Căn cứ:
        eval/results/rr_complement/retrieval_chain_report.md — gold thứ hai của câu đa gold
        đứng hạng dense p50 35 so với 1 của gold lấy được (ghép cặp trong câu, p<1e-4),
        trong khi rng.choice chỉ cho học một gold mỗi lần lấy mẫu."""
        seen, pos = set(), []
        for p in r["pos"]:
            d = p.split("_", 1)[0]
            if p in self.texts and d not in seen:
                seen.add(d); pos.append(p)
        self.rng.shuffle(pos)
        pos = pos[:self.max_pos]
        valid = [1] * len(pos) + [0] * (self.max_pos - len(pos))
        pos = pos + [pos[0]] * (self.max_pos - len(pos))
        negs = [n for n in r["neg"] if n in self.texts]
        negs = (self.rng.sample(negs, self.n_neg) if len(negs) >= self.n_neg
                else negs + [self.rng.choice(negs) for _ in range(self.n_neg - len(negs))])
        sample = (r["query"], [self.texts[p] for p in pos], [self.texts[n] for n in negs], valid)
        return attach_gold(sample, pos + negs, r, self.gold)


def make_collate(tok, q_prefix, d_prefix, q_len, d_len):
    """Gộp query và document vào MỘT batch, encode một lần duy nhất.

    Không phải chuyện tối ưu vặt: gọi model hai lần trong một bước làm DDP báo
    'Expected to mark a variable ready only once' khi bật gradient checkpointing —
    cùng tham số bị đánh dấu sẵn sàng hai lần trong một vòng backward. Gộp lại là
    hết, và tiết kiệm luôn một lượt overhead.
    """
    def fn(batch):
        qs = [q_prefix + b[0] for b in batch]
        docs = []
        multi = isinstance(batch[0][1], list)
        for b in batch:
            if multi:
                docs += [d_prefix + x for x in b[1]]
            else:
                docs.append(d_prefix + b[1])
            docs += [d_prefix + n for n in b[2]]
        # cắt query ngắn trước khi gộp để không bị đệm theo độ dài document
        qs = [tok.decode(tok(x, truncation=True, max_length=q_len,
                             add_special_tokens=False)["input_ids"]) for x in qs]
        enc = tok(qs + docs, padding=True, truncation=True, max_length=d_len,
                  return_tensors="pt")
        known_gold = batch_gold_mask(batch)
        if known_gold is not None:
            enc["known_gold_mask"] = known_gold
        if multi:
            valid = torch.tensor([b[3] for b in batch], dtype=torch.bool)
            return enc, len(batch), len(batch[0][1]) + len(batch[0][2]), valid
        return enc, len(batch), 1 + len(batch[0][2]), None
    return fn


def main(a):
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank = int(os.environ.get("RANK", 0)); local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if ddp:
        dist.init_process_group("nccl"); torch.cuda.set_device(local_rank)
    is_main = rank == 0
    random.seed(a.seed + rank); np.random.seed(a.seed + rank)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    if is_main:
        os.makedirs(a.out_dir, exist_ok=True)
    if ddp:
        dist.barrier()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    logf = open(os.path.join(a.out_dir, f"train_{stamp}.log"), "w", encoding="utf-8") if is_main else None

    def log(m):
        if is_main:
            print(m, flush=True); logf.write(m + "\n"); logf.flush()

    if a.resume_from:
        if not os.path.isfile(os.path.join(a.resume_from, "trainer_state.pt")):
            raise SystemExit(f"resume_from thiếu trainer_state.pt: {a.resume_from}")
        a.model = a.resume_from  # nạp weights từ chính checkpoint resume
    log(f"config: {json.dumps(vars(a), ensure_ascii=False)}")
    a.grad_accum = max(1, a.grad_accum // world)
    log(f"{world} GPU · grad_accum -> {a.grad_accum} · "
        f"batch hiệu dụng {a.batch_size * a.grad_accum * world} query")

    with gzip.open(a.texts, "rt", encoding="utf-8") as f:
        texts = json.load(f)

    tok = AutoTokenizer.from_pretrained(a.model)
    # add_pooling_layer=False: XLM-R có lớp `pooler` (dense+tanh trên token [CLS]) mà
    # mean-pooling không đụng tới. Giữ nó lại thì tham số đó không nhận gradient và DDP
    # chết với "Parameter indices which did not receive grad: 388 389".
    try:
        model = AutoModel.from_pretrained(a.model, add_pooling_layer=False)
    except TypeError:
        model = AutoModel.from_pretrained(a.model)
        if getattr(model, "pooler", None) is not None:
            model.pooler = None
    model.gradient_checkpointing_enable()
    # Lớp word embedding của XLM-R large là 250K x 1024 = 256M/560M tham số. Đóng băng
    # nó cắt gradient + optimizer state gần một nửa (5,6GB -> 4,1GB chi phí cố định) mà
    # gần như không ảnh hưởng chất lượng: fine-tune retriever là chỉnh cách GỘP ngữ
    # nghĩa ở các tầng trên, không phải học lại nghĩa của từng token.
    if a.freeze_embeddings:
        for p_ in model.get_input_embeddings().parameters():
            p_.requires_grad = False
    model.cuda(local_rank)
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"tham số: {n_all:,} tổng · {n_tr:,} huấn luyện"
        + (" (đóng băng embedding)" if a.freeze_embeddings else ""))

    ds = PairData(a.pairs, texts, a.n_neg, a.seed, a.max_pos, getattr(a, "gold_file", None))
    log(f"{len(ds)} cặp huấn luyện")
    sampler = DistributedSampler(ds, shuffle=True, seed=a.seed) if ddp else None
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=(sampler is None), sampler=sampler,
                    num_workers=2, drop_last=True,
                    collate_fn=make_collate(tok, a.q_prefix, a.d_prefix, a.q_len, a.d_len))

    base_steps = len(dl) * a.epochs // a.grad_accum
    total_steps, warmup_steps = base_steps, int(base_steps * a.warmup)
    step, phys_done = 0, 0
    st = None
    if a.resume_from:
        st = torch.load(os.path.join(a.resume_from, "trainer_state.pt"),
                        map_location="cpu", weights_only=False)
        random.setstate(st["rng_python"]); np.random.set_state(st["rng_numpy"])
        torch.set_rng_state(st["rng_torch"]); torch.cuda.set_rng_state_all(st["rng_cuda"])
        ds.rng.setstate(st["ds_rng"])
        step, phys_done = st["global_step"], st["phys_done"]
        if a.extend_horizon:
            total_steps = step + base_steps  # phase mới: giữ momentum, lịch LR mới
            warmup_steps = int(base_steps * a.warmup)
            log(f"resume EXTEND từ {a.resume_from}: step {step}, total mới {total_steps}, "
                f"warmup mới {warmup_steps} (lịch LR restart, giữ momentum optimizer)")
        else:
            if base_steps != st["total_steps"]:
                raise SystemExit(
                    f"horizon khác manifest: manifest total {st['total_steps']} vs "
                    f"invocation hiện tại {base_steps}. Muốn nối thêm epoch với lịch "
                    f"LR mới thì thêm --extend_horizon.")
            total_steps, warmup_steps = st["total_steps"], st["warmup_steps"]
            log(f"resume EXACT từ {a.resume_from}: step {step}/{total_steps}, "
                f"epoch vật lý đã xong {phys_done}")
    params = [p for p in model.parameters() if p.requires_grad]
    if a.optim8bit:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=a.lr, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler()
    if st is not None and not a.extend_horizon:
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        log(f"đã nạp optimizer/scheduler/scaler · lr hiện tại {sched.get_last_lr()[0]:.2e}")
    elif st is not None:
        opt.load_state_dict(st["opt"])  # extend: chỉ giữ momentum, scheduler/scaler mới
        log(f"giữ optimizer state, scheduler mới · lr khởi đầu {sched.get_last_lr()[0]:.2e}")
    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False,
                    static_graph=True)
    log(f"{len(dl)} batch/epoch · {total_steps} bước")

    if a.reserve_gb > 0:
        from eval.gpu_reserve import reserve_vram
        reserve_vram(a.reserve_gb, local_rank, log)

    t0 = time.time(); rl = ra = 0.0; seen = 0
    for ep in range(a.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(ep)
        for it, (enc, nq, per_q, valid) in enumerate(dl):
            known_gold = enc.pop("known_gold_mask", None)
            if known_gold is not None:
                known_gold = known_gold.cuda(local_rank, non_blocking=True)
            enc = {k: v.cuda(local_rank, non_blocking=True) for k, v in enc.items()}
            with torch.cuda.amp.autocast(dtype=torch.float16):
                emb = F.normalize(mean_pool(model(**enc), enc["attention_mask"], a.pooling), dim=-1)
                qv, dv = emb[:nq], emb[nq:]
                if valid is None:
                    # cột 0..nq-1 là positive của từng query (in-batch negative);
                    # phần còn lại là hard negative, dùng chung cho cả batch.
                    pos_idx = torch.arange(nq, device=dv.device) * per_q
                    pos = dv[pos_idx]
                    hard = dv[[i for i in range(dv.shape[0]) if i % per_q != 0]]
                    cand = torch.cat([pos, hard], 0)
                    logits = (qv @ cand.T) / a.temperature
                    target = torch.arange(nq, device=logits.device)
                    if known_gold is not None:
                        order = list(range(0, dv.shape[0], per_q)) + [
                            i for i in range(dv.shape[0]) if i % per_q != 0]
                        logits = mask_known_gold(logits, known_gold[:, order], target)
                    loss = F.cross_entropy(logits, target)
                else:
                    # nhiều positive: mỗi positive hợp lệ của query i phải thắng MỌI ứng viên
                    # không thuộc positive của chính i (positive cùng câu không làm negative
                    # của nhau; positive câu khác + hard negative vẫn là negative như cũ).
                    P = a.max_pos
                    lg = (qv @ dv.T).float() / a.temperature          # nq x nq*per_q
                    own = torch.zeros_like(lg, dtype=torch.bool)
                    ar = torch.arange(nq, device=lg.device)
                    for j in range(P):
                        own[ar, ar * per_q + j] = True
                    blocked = own if known_gold is None else own | known_gold
                    neg_lse = lg.masked_fill(blocked, float("-inf")).logsumexp(1) # nq
                    sp = torch.stack([lg[ar, ar * per_q + j] for j in range(P)], 1)  # nq x P
                    vm = valid.to(lg.device)
                    lp = -sp + torch.logaddexp(sp, neg_lse[:, None])
                    loss = lp[vm].mean()
                    logits = lg
                    target = ar * per_q      # acc: top-1 là positive đầu tiên (chỉ để log)
            sync = (it + 1) % a.grad_accum == 0
            with (model.no_sync() if (ddp and not sync) else _nullcontext()):
                scaler.scale(loss / a.grad_accum).backward()
            rl += loss.item(); ra += (logits.argmax(1) == target).float().mean().item(); seen += 1
            if sync:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                sched.step(); step += 1
                if step % a.log_every == 0:
                    log(f"ep {ep} step {step}/{total_steps} · loss {rl/seen:.4f} · acc {ra/seen:.4f} · "
                        f"lr {sched.get_last_lr()[0]:.2e} · {(time.time()-t0)/60:.1f} phút")
                    rl = ra = 0.0; seen = 0
        if is_main:
            phys = (phys_done + ep + 1) if st is not None else (a.epoch_offset + ep + 1)
            ck = os.path.join(a.out_dir, f"epoch{phys}")
            (model.module if ddp else model).save_pretrained(ck); tok.save_pretrained(ck)
            state = {"phys_done": phys, "global_step": step, "total_steps": total_steps,
                     "warmup_steps": warmup_steps, "opt": opt.state_dict(),
                     "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                     "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                     "rng_torch": torch.get_rng_state(),
                     "rng_cuda": torch.cuda.get_rng_state_all(), "ds_rng": ds.rng.getstate(),
                     "seed": a.seed, "lr": a.lr}
            tmp = os.path.join(ck, ".trainer_state.tmp.pt")
            torch.save(state, tmp)
            os.replace(tmp, os.path.join(ck, "trainer_state.pt"))
            json.dump({"last_ckpt": ck, "phys_done": phys, "global_step": step,
                       "total_steps": total_steps, "config": vars(a)},
                      open(os.path.join(a.out_dir, "training_manifest.json"), "w",
                           encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            log(f"đã lưu {ck} + trainer_state (step {step}/{total_steps})")
        if ddp:
            dist.barrier()
    log(f"xong sau {(time.time()-t0)/60:.1f} phút")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="intfloat/multilingual-e5-large")
    p.add_argument("--pairs", default="eval/train_pairs/legalir_pairs_r2.jsonl")
    p.add_argument("--gold_file", default=None,
                   help="nhãn của train split để loại known-gold khỏi negative chéo batch; "
                        "bỏ cờ để tái lập loss cũ, không dùng nhãn dev")
    p.add_argument("--texts", default="retrieval/db_fast/texts.json.gz")
    p.add_argument("--out_dir", default="eval/ckpt/e5_ft")
    p.add_argument("--q_prefix", default="query: ")
    p.add_argument("--d_prefix", default="passage: ")
    p.add_argument("--q_len", type=int, default=64)
    p.add_argument("--d_len", type=int, default=384)
    p.add_argument("--n_neg", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--optim8bit", action="store_true")
    p.add_argument("--reserve_gb", type=float, default=0.0,
                   help="giữ trước VRAM tới mức đỉnh, chống bị chen giữa chừng")
    p.add_argument("--freeze_embeddings", action="store_true", default=True)
    p.add_argument("--no_freeze_embeddings", dest="freeze_embeddings",
                   action="store_false")
    p.add_argument("--pooling", choices=["mean","cls"], default="mean")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_pos", type=int, default=1,
                   help="1 = hành vi cũ (1 positive ngẫu nhiên). >1 = loss nhiều positive")
    p.add_argument("--epoch_offset", type=int, default=0,
                   help="epoch sweep (09/2026): cộng vào tên checkpoint vật lý, "
                   "VD --epochs 2 --epoch_offset 2 -> epoch3/epoch4. Bị bỏ qua khi "
                   "--resume_from (tên nối từ manifest).")
    p.add_argument("--resume_from", default=None,
                   help="thư mục checkpoint epoch có trainer_state.pt để resume. "
                   "Mặc định resume EXACT (cùng horizon, nối trajectory); thêm "
                   "--extend_horizon để nối thêm epoch với lịch LR mới (giữ momentum).")
    p.add_argument("--extend_horizon", action="store_true",
                   help="mở rộng horizon: total mới = step đã xong + step còn lại, "
                   "scheduler/scaler mới, chỉ giữ optimizer state.")
    main(p.parse_args())
