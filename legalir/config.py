"""Nạp cấu hình YAML (configs/fam.yaml — cấu hình duy nhất của gói).

Mỗi cấu hình = một bộ checkpoint + độ sâu truy hồi, đúng biến của configs/*.sh trước đây
(chỉ chuyển sang YAML, không đổi giá trị nào).

    from legalir.config import load_config
    cfg = load_config("fam", configs_dir="configs")
"""
import os

import yaml


class Config:
    """Bộ tham số của một cấu hình suy luận."""

    def __init__(self, d, name):
        self.name = name
        self.dense_model = d["dense_model"]          # kênh bge (vnembv2 FT, CLS)
        self.dense_prefix = d["dense_prefix"]        # ma trận dense của kênh bge
        self.e5_model = d["e5_model"]                # kênh e5 (harrier FT, last-token)
        self.e5_prefix = d["e5_prefix"]              # ma trận dense của kênh e5
        self.ce_model = d["ce_model"]                # cross-encoder chấm chunk
        self.topk_bm25 = int(d["topk_bm25"])         # độ sâu truy hồi kênh bm25
        self.topk_dense = int(d["topk_dense"])       # độ sâu truy hồi 3 kênh còn lại
        self.reference = d.get("reference", "")      # bài public đã nộp (đối chứng team)

    def __repr__(self):
        return (f"Config({self.name}: dense={os.path.basename(self.dense_model)} · "
                f"e5={os.path.basename(self.e5_model)} · ce={os.path.basename(self.ce_model)} · "
                f"topk={self.topk_bm25}/{self.topk_dense})")


def load_config(name, configs_dir="configs"):
    """Nạp configs/<name>.yaml — đường dẫn tương đối tính từ gói (chứa configs/)."""
    path = name if name.endswith(".yaml") else os.path.join(configs_dir, f"{name}.yaml")
    if not os.path.isfile(path):
        raise SystemExit(f"⛔ không có cấu hình {path}")
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return Config(d, d.get("name", os.path.splitext(os.path.basename(path))[0]))
