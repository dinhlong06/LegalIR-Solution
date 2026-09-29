# Vendored verbatim từ eval/gold_negative_mask.py (repo dev) — không đổi một dòng
# logic, chỉ thêm comment này (ghi provenance). Dùng bởi train_biencoder*.py qua
# `from .gold_negative_mask import ...` (giữ nguyên cơ chế relative import).
"""Loại chunk của document gold đã biết khỏi negative contrastive.

Không tự coi mọi chunk của gold là positive. Target được chọn bởi miner giữ nguyên.
Kích hoạt bằng --gold_file trỏ đúng nhãn của TRAIN SPLIT; bỏ cờ để tái lập cách cũ.
"""
import json
import torch


def load_gold_file(path, rows):
    if path is None:
        return None
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    gold = {q: set(map(str, r['answer'] if isinstance(r, dict) else r)) for q, r in raw.items()}
    for r in rows:
        if r['qid'] not in gold or not gold[r['qid']]:
            raise ValueError(f"gold_file thiếu nhãn train cho {r['qid']}")
        if any(c.split('_', 1)[0] not in gold[r['qid']] for c in r['pos']):
            raise ValueError(f"positive không thuộc gold_file: {r['qid']}")
    return gold


def attach_gold(sample, chunk_ids, row, gold):
    if gold is None:
        return sample
    return sample + ({'gold': sorted(gold[row['qid']]),
                      'doc_ids': [c.split('_', 1)[0] for c in chunk_ids]},)


def batch_gold_mask(batch):
    if not isinstance(batch[0][-1], dict):
        return None
    docs = [d for row in batch for d in row[-1]['doc_ids']]
    return torch.tensor([[d in set(row[-1]['gold']) for d in docs] for row in batch], dtype=torch.bool)


def mask_known_gold(logits, known_gold, targets):
    """Chỉ mask negative của single-positive CE; khôi phục target chính xác."""
    blocked = known_gold.clone()
    blocked[torch.arange(logits.shape[0], device=logits.device), targets] = False
    return logits.masked_fill(blocked, float('-inf'))
