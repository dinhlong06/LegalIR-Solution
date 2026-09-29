"""Fine-tune cross-encoder reranker trên dữ liệu BTC. Chỉ train.json, không nguồn ngoài.

Hàm mất mát: softmax theo NHÓM (listwise), không phải binary cross-entropy từng cặp.
Mỗi nhóm = 1 positive + N negative của cùng một câu hỏi; model học xếp positive lên
đầu NHÓM ĐÓ. Đây đúng việc nó phải làm lúc inference (xếp hạng trong pool ~93 chunk),
trong khi BCE từng cặp lại bắt nó học một ngưỡng tuyệt đối vô nghĩa.

Ràng buộc phần cứng: RTX 2080 Ti là Turing — KHÔNG có bf16, phải fp16 + GradScaler.
Không chạy được FlashAttention-2. 11GB dùng chung nên đóng băng lớp word embedding
(256M/568M tham số của XLM-R nằm ở đó) để cắt 2/3 bộ nhớ optimizer.

Tái lập (rule.md §4): cố định seed, ghi log không đè, lưu config cùng checkpoint.
"""
import argparse, gzip, json, math, os, random, sys, time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from contextlib import nullcontext as _nullcontext

from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


class GroupDataset(Dataset):
    """Mỗi nhóm = |P| positive + n_neg negative của cùng một câu hỏi.

    `max_pos = 1` tái lập ĐÚNG hành vi cũ (lấy ngẫu nhiên 1 positive mỗi epoch —
    chính là biến thể `Rand1LH` mà KDD'26 đo là yếu nhất). `max_pos > 1` giữ nhiều
    gold cùng lúc để loss có thể yêu cầu model đẩy CẢ HAI lên đầu — 7,9% câu train
    là multi-gold nhưng gánh 43% điểm mất (fix_plan.md §1).
    """

    def __init__(self, path, texts, n_neg, seed, max_pos=1):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.texts, self.n_neg, self.max_pos = texts, n_neg, max_pos
        self.rng = random.Random(seed)
        self.rows = [r for r in self.rows
                     if any(p in texts for p in r["pos"]) and
                     sum(n in texts for n in r["neg"]) >= 1]
        npos = [min(sum(p in texts for p in r["pos"]), max_pos) for r in self.rows]
        print(f"{len(self.rows)} nhóm huấn luyện · "
              f"{sum(n > 1 for n in npos)} nhóm có >1 positive · "
              f"tối đa {max(npos)} positive/nhóm")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        avail = [p for p in r["pos"] if p in self.texts]
        if len(avail) > self.max_pos:
            avail = self.rng.sample(avail, self.max_pos)
        negs = [n for n in r["neg"] if n in self.texts]
        if len(negs) >= self.n_neg:
            negs = self.rng.sample(negs, self.n_neg)
        else:
            negs = negs + [self.rng.choice(negs) for _ in range(self.n_neg - len(negs))]
        return (r["query"],
                [self.texts[p] for p in avail] + [self.texts[n] for n in negs],
                len(avail))


class DocGroupDataset(Dataset):
    """Nhom o CAP VAN BAN (eval/mine_doc_groups.py): groups[0] = gold doc, con lai negative.

    Hai che do dung CHUNG mot file, chi khac CACH TINH LOSS — do la bien duy nhat:
      loss=listwise : lay DUNG 1 chunk gold (cai CE chAm cao nhat) + moi chunk negative
                      => dung dang loss cu, chi doi tap negative  (arm doi chung DATA)
      loss=doclse   : giu moi chunk, tra ve ranh gioi tung VAN BAN => loss gop LSE roi
                      tinh cross-entropy tren DOCUMENT           (arm doi chung LOSS)
    """

    def __init__(self, path, texts, mode, seed, max_chunk=2, concat=False):
        self.rows = []
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            g = [[c for c in grp if c in texts][:max_chunk] for grp in r["groups"]]
            if not g[0] or sum(len(x) > 0 for x in g[1:]) < 2:
                continue
            r["groups"] = [g[0]] + [x for x in g[1:] if x]
            self.rows.append(r)
        self.texts, self.mode, self.concat = texts, mode, concat
        self.rng = random.Random(seed)
        nd = [len(r["groups"]) for r in self.rows]
        npair = [sum(len(x) for x in r["groups"]) for r in self.rows]
        print(f"{len(self.rows)} nhom cap van ban · {np.mean(nd):.1f} van ban/nhom · "
              f"{np.mean(npair):.1f} cap/nhom")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        g = r["groups"]
        if self.concat:
            # MOT text moi VAN BAN = noi cac chunk evidence => model doc chung ca hai
            # trong MOT forward (cross-attention giua cac evidence), roi CE tren document.
            return (r["query"], [" ".join(self.texts[c] for c in grp) for grp in g], 1)
        if self.mode == "doclse":
            flat = [c for grp in g for c in grp]
            return (r["query"], [self.texts[c] for c in flat], [len(x) for x in g])
        flat = [g[0][0]] + [c for grp in g[1:] for c in grp]
        return (r["query"], [self.texts[c] for c in flat], 1)


def collate(batch, tok, max_len):
    pairs = [(q, p) for q, ps, _ in batch for p in ps]
    enc = tok([a for a, _ in pairs], [b for _, b in pairs],
              padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    # Nhóm dài ngắn khác nhau (|P| thay đổi) ⇒ không view được thành ma trận đều,
    # phải trả kèm ranh giới từng nhóm để tách logits sau forward.
    return enc, [len(ps) for _, ps, _ in batch], [np for _, _, np in batch]


BCE_WEIGHT = 0.5


def group_loss(logits, sizes, n_pos, mode):
    """logits: [tổng số cặp] phẳng. Trả (loss trung bình theo nhóm, acc@1).

    `lsepair` (KDD'26, arXiv 2602.12727): L = log(1 + Σ_{p∈P} Σ_{n∈N} exp(s_n - s_p)).
    Phạt MỌI cặp negative đứng trên positive ⇒ mọi gold đều phải lên đầu, không chỉ
    một cái được bốc ngẫu nhiên.

    Với |P| = 1 nó bằng ĐÚNG cross_entropy(logits, 0) của bản cũ — dùng làm phép
    kiểm tái lập trước khi tin số nào.
    """
    logits = logits.float()
    losses, accs, off = [], [], 0
    for sz, npz in zip(sizes, n_pos):
        g = logits[off:off + sz]
        off += sz
        if mode == "doclse":
            # npz la DANH SACH so chunk cua tung van ban; van ban 0 la gold.
            # s_doc = T*logsumexp(logit/T) — DUNG cong thuc gop luc suy luan (T=2).
            T = 2.0
            ds, o = [], 0
            for n in npz:
                ds.append(T * torch.logsumexp(g[o:o + n] / T, 0))
                o += n
            d = torch.stack(ds)
            losses.append(F.cross_entropy(d.unsqueeze(0),
                                          torch.zeros(1, dtype=torch.long, device=d.device)))
            accs.append(float(d.argmax().item() == 0))
            continue
        s_pos, s_neg = g[:npz], g[npz:]
        if mode in ("bce", "listwise_bce"):
            # Listwise softmax chi rang buoc HIEU giua cac ung vien trong nhom, muc tuyet doi
            # troi tu do. Bang chung 12/09: checkpoint `fullpos_l1` (listwise) xuat diem bao
            # hoa min 0,9966 · std 0,0001 => fp16 mat sach do phan giai, recall sup 6,93.
            # BCE neo diem TUYET DOI: positive -> 1, negative -> 0.
            tgt = torch.zeros_like(g); tgt[:npz] = 1.0
            bce = F.binary_cross_entropy_with_logits(g, tgt)
            if mode == "bce":
                losses.append(bce)
            else:                                    # giu muc tieu xep hang da biet la tot
                lw = F.cross_entropy(g.unsqueeze(0),
                                     torch.zeros(1, dtype=torch.long, device=g.device))
                losses.append(lw + BCE_WEIGHT * bce)
        elif mode == "listwise":                     # bản cũ: 1 positive vs phần còn lại
            losses.append(F.cross_entropy(g.unsqueeze(0),
                                          torch.zeros(1, dtype=torch.long,
                                                      device=g.device)))
        else:                                        # lsepair
            diff = s_neg.unsqueeze(0) - s_pos.unsqueeze(1)      # [|P|, |N|]
            losses.append(F.softplus(torch.logsumexp(diff.reshape(-1), 0)))
        accs.append(float(g.argmax().item() < npz))
    return torch.stack(losses).mean(), sum(accs) / len(accs)


def main(a):
    global BCE_WEIGHT
    BCE_WEIGHT = getattr(a, 'bce_weight', 0.5)
    # DDP: mỗi GPU giữ một bản model đầy đủ, chia dữ liệu, đồng bộ gradient.
    # Batch hiệu dụng = groups_per_batch × grad_accum × world_size, nên phải chia
    # grad_accum cho world_size để giữ nguyên batch hiệu dụng khi tăng số GPU —
    # nếu không thì đổi số GPU là đổi luôn siêu tham số, không so sánh được.
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    is_main = rank == 0

    random.seed(a.seed + rank); np.random.seed(a.seed + rank)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    if is_main:
        os.makedirs(a.out_dir, exist_ok=True)
    if ddp:
        dist.barrier()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    logf = open(os.path.join(a.out_dir, f"train_{stamp}.log"), "w", encoding="utf-8") if is_main else None

    def log(msg):
        if not is_main:
            return
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    a.grad_accum = max(1, a.grad_accum // world)
    log(f"DDP: {world} GPU · grad_accum điều chỉnh -> {a.grad_accum} "
        f"(batch hiệu dụng {a.groups_per_batch * a.grad_accum * world} nhóm)")

    log(f"config: {json.dumps(vars(a), ensure_ascii=False)}")
    log(f"torch {torch.__version__} · gpu {torch.cuda.get_device_name(0)} · "
        f"capability {torch.cuda.get_device_capability(0)} (bf16 = "
        f"{torch.cuda.is_bf16_supported()})")

    with gzip.open(a.texts, "rt", encoding="utf-8") as f:
        texts = json.load(f)
    log(f"{len(texts)} chunk text")

    # trust_remote_code chỉ bật khi được yêu cầu rõ ràng: mọi arm cũ (XLM-R) giữ
    # nguyên đường đi cũ từng bit. Cần cho họ kiến trúc lạ như gte-multilingual.
    trc = {"trust_remote_code": True} if a.trust_remote_code else {}
    tok = AutoTokenizer.from_pretrained(a.model, **trc)
    # Qwen3 khong co pad token; XLM-R co san nen nhanh nay khong dung toi arm cu.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # LoRA + base fp16: full fine-tune cần ~4,2GB chi phí cố định (model fp32 2,27
    # + gradient 1,25 + optimizer state) trước khi tính activation — không vừa GPU
    # dùng chung. LoRA hạ xuống ~1,2GB. Hợp lệ theo rule.md §2.1: cấm LoRA trên
    # model gốc >4B rồi khai <4B, còn base ở đây là 567,8M nên không vướng.
    dtype = torch.float16 if a.lora else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=1, torch_dtype=dtype, **trc)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if a.lora:
        from peft import LoraConfig, get_peft_model
        # Ten module khac nhau giua ho kien truc: XLM-R dung query/key/value/dense,
        # Qwen3 dung q_proj/k_proj/... Dat sai ten thi peft bao loi hoac gan 0 lop.
        targets = a.lora_targets or ["query", "key", "value", "dense"]
        cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05,
                         bias="none", task_type="SEQ_CLS",
                         target_modules=targets)
        model = get_peft_model(model, cfg)
        for n_, p_ in model.named_parameters():           # head phân loại phải học
            if "classifier" in n_ or "score" in n_:
                p_.requires_grad = True
                p_.data = p_.data.float()
        for n_, p_ in model.named_parameters():
            if p_.requires_grad and p_.dtype == torch.float16:
                p_.data = p_.data.float()
    else:
        for p_ in model.get_input_embeddings().parameters():
            p_.requires_grad = False
    model.cuda(local_rank)

    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"tham số: {n_all:,} tổng · {n_tr:,} huấn luyện "
        f"({'LoRA r=%d' % a.lora_r if a.lora else 'full, đóng băng embedding'})")

    if a.docgroups:
        ds = DocGroupDataset(a.docgroups, texts, a.loss, a.seed, a.max_chunk, a.concat_doc)
    else:
        ds = GroupDataset(a.pairs, texts, a.n_neg, a.seed, a.max_pos)
    sampler = DistributedSampler(ds, shuffle=True, seed=a.seed) if ddp else None
    dl = DataLoader(ds, batch_size=a.groups_per_batch, shuffle=(sampler is None),
                    sampler=sampler, num_workers=2,
                    collate_fn=lambda b: collate(b, tok, a.max_length), drop_last=True)

    steps = len(dl) * a.epochs // a.grad_accum
    params = [p for p in model.parameters() if p.requires_grad]
    if a.optim8bit:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=a.lr, weight_decay=0.01)
        log("optimizer: AdamW 8-bit (tiết kiệm ~2GB state)")
    else:
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    sched = get_linear_schedule_with_warmup(opt, int(steps * a.warmup), steps)
    scaler = torch.cuda.amp.GradScaler()
    log(f"{len(dl)} batch/epoch · {steps} bước cập nhật")

    if a.reserve_gb > 0:
        # Giu cho VRAM den muc DINH ngay tu dau, de tien trinh khac tren may dung chung
        # khong chen vao roi lam ta OOM luc phinh dan len dinh (~9,4GB).
        #
        # Caching allocator cua PyTorch khong tra bo nho da cap phat ve driver (mien la
        # KHONG bat expandable_segments), nen khoi nay sau khi `del` van nam trong cache:
        # nguoi khac khong lay duoc, con ta thi tai su dung chinh no cho activation.
        #
        # PHAI dat SAU khi model da len GPU: `from_pretrained` (qua accelerate) co goi
        # torch.cuda.empty_cache(), dat truoc do la bi xoa sach — da dinh that, log bao
        # giu 10GB nhung nvidia-smi chi thay 5,8GB.
        free_b, _ = torch.cuda.mem_get_info(local_rank)
        want = min(a.reserve_gb, free_b / 2 ** 30 - 0.4)     # chua 0,4GB cho context
        chunks, per = [], int(0.25 * (1024 ** 3) / 2)        # khoi 0,25GB fp16
        try:
            for _ in range(max(0, int(want / 0.25))):
                chunks.append(torch.empty(per, dtype=torch.float16,
                                          device=f"cuda:{local_rank}"))
        except torch.cuda.OutOfMemoryError:
            pass
        got = len(chunks) * 0.25
        del chunks
        log(f"giữ chỗ VRAM: {got:.2f} GB · PyTorch reserved "
            f"{torch.cuda.memory_reserved(local_rank)/2**30:.2f} GB")

    step = 0; t0 = time.time(); run_loss = 0.0; run_acc = 0.0; seen = 0
    for ep in range(a.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(ep)
        for it, (enc, sizes, n_pos) in enumerate(dl):
            enc = {k: v.cuda(local_rank, non_blocking=True) for k, v in enc.items()}
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(**enc).logits.view(-1)
                loss, acc = group_loss(logits, sizes, n_pos, a.loss)
            sync = (it + 1) % a.grad_accum == 0
            ctx = model.no_sync() if (ddp and not sync) else _nullcontext()
            with ctx:
                scaler.scale(loss / a.grad_accum).backward()
            run_loss += loss.item()
            run_acc += acc
            seen += 1
            if (it + 1) % a.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                sched.step(); step += 1
                if step % a.log_every == 0:
                    log(f"ep {ep} step {step}/{steps} · loss {run_loss/seen:.4f} · "
                        f"acc@1 trong nhóm {run_acc/seen:.4f} · lr {sched.get_last_lr()[0]:.2e} · "
                        f"{(time.time()-t0)/60:.1f} phút")
                    run_loss = run_acc = 0.0; seen = 0
        if is_main:
            ckpt = os.path.join(a.out_dir, f"epoch{ep+1}")
            core = model.module if ddp else model
            if a.lora:
                # LƯU ADAPTER, KHÔNG merge tại đây. `merge_and_unload()` sửa model TẠI
                # CHỖ và trả về model nền: gọi nó sau epoch 1 thì epoch 2 train trên
                # model ĐÃ MẤT LoRA, và `.half()` ép tham số huấn luyện về fp16 nên
                # `scaler.unscale_` ném "Attempting to unscale FP16 gradients" ngay ở
                # bước optimizer đầu tiên của epoch 2. Đã cắn thật (arm q3vlsp/q3base,
                # 06/09) — cùng lỗi mà train_qwen3_reranker.py đã cảnh báo sẵn. Việc gộp
                # dời xuống sau vòng lặp, trên bản nạp lại từ đĩa.
                core.save_pretrained(ckpt)
            else:
                core.save_pretrained(ckpt)
            tok.save_pretrained(ckpt)
            log(f"đã lưu {ckpt}")
            # Dinh CAP PHAT THAT. `memory_reserved` khong dung duoc: phan giu cho nam
            # trong cache nen no do "ta dang om bao nhieu", khong phai "ta CAN bao nhieu".
            # Con so nay moi la can cu dat --reserve_gb va nguong cho lan sau.
            log(f"đỉnh cấp phát thật: "
                f"{torch.cuda.max_memory_allocated()/2**30:.2f} GB")
        if ddp:
            dist.barrier()
    log(f"xong sau {(time.time()-t0)/60:.1f} phút")

    # ---- gộp adapter thành checkpoint đầy đủ: SAU vòng lặp, NẠP LẠI TỪ ĐĨA ----
    # Nạp lại thay vì dùng `model` đang ở RAM: merge_and_unload() phá huỷ đối tượng,
    # làm trên bản trong RAM là lặp lại đúng lỗi vừa sửa ở trên. Ghi ra `epochN_merged`
    # để score_pairs_hf nạp thẳng bằng AutoModelForSequenceClassification, không cần peft.
    if a.lora and is_main:
        try:
            del model
        except NameError:
            pass
        torch.cuda.empty_cache()
        from peft import PeftModel
        for ep_ in range(a.epochs):
            ad = os.path.join(a.out_dir, f"epoch{ep_+1}")
            mg = ad + "_merged"
            if not os.path.isdir(ad) or os.path.isdir(mg):
                continue
            base_ = AutoModelForSequenceClassification.from_pretrained(
                a.model, num_labels=1, torch_dtype=torch.float16, **trc)
            if getattr(base_.config, "pad_token_id", None) is None:
                base_.config.pad_token_id = tok.pad_token_id
            merged_ = PeftModel.from_pretrained(base_, ad).merge_and_unload()
            merged_.save_pretrained(mg); tok.save_pretrained(mg)
            del base_, merged_
            log(f"đã gộp {mg}")

    if is_main:
        logf.close()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="AITeamVN/Vietnamese_Reranker")
    p.add_argument("--pairs", default="eval/train_pairs/legalir_pairs.jsonl")
    p.add_argument("--texts", default="retrieval/db_fast/texts.json.gz")
    p.add_argument("--out_dir", default="eval/ckpt/vn_reranker_ft")
    p.add_argument("--n_neg", type=int, default=7)
    # max_pos=1 + loss=listwise  ⇒ tái lập ĐÚNG bản `fam` đã nộp (0,9428).
    p.add_argument("--max_pos", type=int, default=1,
                   help="số positive tối đa mỗi nhóm; 1 = hành vi cũ (Rand1LH)")
    p.add_argument("--loss", choices=["listwise", "lsepair", "doclse", "bce", "listwise_bce"], default="listwise")
    p.add_argument("--bce_weight", type=float, default=0.5,
                   help="listwise_bce: L = listwise + bce_weight * BCE")
    p.add_argument("--docgroups", help="file nhom cap van ban (eval/mine_doc_groups.py)")
    p.add_argument("--max_chunk", type=int, default=2)
    p.add_argument("--concat_doc", action="store_true",
                   help="noi cac chunk cua mot van ban thanh MOT text (de xuat §2-A3)")
    p.add_argument("--groups_per_batch", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reserve_gb", type=float, default=0.0,
                   help="giữ trước ngần này GB VRAM ngay lúc khởi động, để tiến trình "
                        "khác không chen vào giữa chừng làm OOM khi ta chạm đỉnh")
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_targets", nargs="*", default=None,
                   help="ten module cho LoRA; mac dinh la ten XLM-R")
    p.add_argument("--trust_remote_code", action="store_true",
                   help="cho model có code riêng trên HF (gte-multilingual...)")
    p.add_argument("--optim8bit", action="store_true",
                   help="AdamW 8-bit: state 0,6GB thay vì 2,5GB, vừa được GPU trống ít")
    main(p.parse_args())
