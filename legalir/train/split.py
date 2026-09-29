# Gộp từ eval/make_dev_split.py + eval/make_split_v2.py (repo dev), giữ nguyên logic —
# chỉ thay đổi: đường dẫn thành tham số (mặc định theo layout gói), hai main() gộp
# thành build_v1()/build_v2() + một CLI duy nhất. Bản gốc nằm ở
# training_logs/original_scripts/ (không có ở đây — hai file này thuộc repo dev).
"""Tách train/dev cho fine-tune LegalIR.

v1 (make_dev_split): cắt dev theo VĂN BẢN (document-disjoint) — vì một gold doc có thể
trả lời tới 109 câu, chia theo câu hỏi sẽ làm dev/train dùng chung gold doc và thổi
phồng mọi kết quả fine-tune. Dựng đồ thị hai phía (câu hỏi <-> gold doc), rải các thành
phần liên thông vào dev/train.

v2 (make_split_v2): sửa lỗi của v1 — dev cũ loại sạch các cụm 'hub' nên dễ hơn thật.
Giữ nguyên tính tách rời theo văn bản bằng cách chia theo thành phần liên thông, nhưng
phân bổ hai cụm hub về hai phía: một cụm vào dev, một cụm vào train. Dev nặng hub hơn
thực tế (44% so với 23%), nên xuất kèm TRỌNG SỐ tái cân để quy đổi về phân phối thật.
"""
import argparse
import json
import os
import random


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, p):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def connected_components(train):
    """Trả về list các (qids, docids) là thành phần liên thông của đồ thị q<->doc."""
    doc2q = {}
    for qid, item in train.items():
        for d in item["answer"]:
            doc2q.setdefault(str(d), []).append(qid)

    seen_q, seen_d, comps = set(), set(), []
    for qid in train:
        if qid in seen_q:
            continue
        stack_q, stack_d = [qid], []
        comp_q, comp_d = set(), set()
        while stack_q or stack_d:
            if stack_q:
                q = stack_q.pop()
                if q in comp_q:
                    continue
                comp_q.add(q)
                for d in train[q]["answer"]:
                    d = str(d)
                    if d not in comp_d:
                        stack_d.append(d)
            else:
                d = stack_d.pop()
                if d in comp_d:
                    continue
                comp_d.add(d)
                for q in doc2q.get(d, []):
                    if q not in comp_q:
                        stack_q.append(q)
        seen_q |= comp_q
        seen_d |= comp_d
        comps.append((sorted(comp_q), sorted(comp_d)))
    return comps


def build_v1(train_file, out_dir, dev_size=1000, max_comp=60, seed=42):
    """Cắt dev v1: rải thành phần liên thông (bỏ cụm > max_comp), seed cố định."""
    random.seed(seed)
    train = load_json(train_file)
    print(f"train.json: {len(train)} câu hỏi")

    comps = connected_components(train)
    print(f"Số thành phần liên thông: {len(comps)}")
    sizes = sorted((len(q) for q, _ in comps), reverse=True)
    print(f"  thành phần lớn nhất: {sizes[0]} câu · top-5: {sizes[:5]}")

    # Hai thành phần khổng lồ (819 và 793 câu) là các cụm hub nối với nhau qua câu
    # nhiều gold doc. Nuốt trọn một cụm vào dev sẽ làm dev lệch hẳn: tỷ lệ câu
    # nhiều gold vọt lên ~15% trong khi toàn train chỉ 7,9%. Chặn theo kích thước
    # để dev gồm nhiều cụm nhỏ, bám sát phân phối thật.
    pool = [c for c in comps if len(c[0]) <= max_comp]
    big = [c for c in comps if len(c[0]) > max_comp]
    print(f"  loại {len(big)} thành phần > {max_comp} câu "
          f"({sum(len(q) for q, _ in big)} câu) khỏi vùng chọn dev")

    random.shuffle(pool)
    dev_q, dev_d = [], set()
    for qids, docids in pool:
        if len(dev_q) >= dev_size:
            break
        dev_q.extend(qids)
        dev_d |= set(docids)

    dev_set = set(dev_q)
    train_q = [q for q in train if q not in dev_set]

    # kiểm tra bất biến: không gold doc nào bị chia sẻ
    train_d = set()
    for q in train_q:
        train_d |= {str(d) for d in train[q]["answer"]}
    overlap = dev_d & train_d
    assert not overlap, f"RÒ RỈ: {len(overlap)} gold doc nằm ở cả hai split"

    dev_q = sorted(dev_q)
    save_json({q: {"question": train[q]["question"]} for q in dev_q},
              os.path.join(out_dir, "dev_questions.json"))
    save_json({q: [str(d) for d in train[q]["answer"]] for q in dev_q},
              os.path.join(out_dir, "dev_gt.json"))
    save_json(sorted(train_q), os.path.join(out_dir, "train_ids.json"))

    n_multi = sum(1 for q in dev_q if len(train[q]["answer"]) > 1)
    print(f"\ndev : {len(dev_q)} câu · {len(dev_d)} gold doc · {n_multi} câu nhiều gold")
    print(f"train: {len(train_q)} câu · {len(train_d)} gold doc")
    print(f"giao nhau gold doc: {len(overlap)} ✅")
    print(f"Đã ghi vào {out_dir}/")


def build_v2(train_file, v1_dev_gt, out_dir, big=60):
    """Cắt dev v2 từ train.json + dev_gt của v1: giữ dev v1, phân 2 cụm hub về 2 phía."""
    tr = load_json(train_file)
    comps = connected_components(tr)
    bigc = sorted([c for c in comps if len(c[0]) > big], key=lambda c: -len(c[0]))
    print(f"{len(bigc)} cụm hub: {[len(c[0]) for c in bigc]}")

    dev_v1 = set(load_json(v1_dev_gt))                 # 1.001 câu cụm nhỏ, giữ nguyên
    hub_dev = bigc[1]                                  # cụm hub thứ hai -> dev
    hub_train = bigc[0]                                # cụm hub thứ nhất -> train
    _ = hub_train                                      # giữ trong train theo thiết kế

    dev_q = sorted(dev_v1 | set(hub_dev[0]))
    train_q = sorted(set(tr) - set(dev_q))

    dev_d = {str(d) for q in dev_q for d in tr[q]["answer"]}
    train_d = {str(d) for q in train_q for d in tr[q]["answer"]}
    assert not (dev_d & train_d), f"RÒ RỈ {len(dev_d & train_d)} gold doc"

    hub_all = {q for c in bigc for q in c[0]}
    w_hub_true = len(hub_all) / len(tr)
    n_hub_dev = len(set(dev_q) & hub_all)
    print(f"\ndev  : {len(dev_q)} câu ({n_hub_dev} hub = {n_hub_dev/len(dev_q):.1%}) · {len(dev_d)} gold doc")
    print(f"train: {len(train_q)} câu · {len(train_d)} gold doc · rò rỉ 0 ✅")
    print(f"tỉ lệ hub THẬT của train: {w_hub_true:.1%}")

    save_json({q: {"question": tr[q]["question"]} for q in dev_q}, f"{out_dir}/dev_questions.json")
    save_json({q: [str(d) for d in tr[q]["answer"]] for q in dev_q}, f"{out_dir}/dev_gt.json")
    save_json({q: {"question": tr[q]["question"], "answer": [str(d) for d in tr[q]["answer"]]}
               for q in train_q}, f"{out_dir}/train.json")
    save_json({"hub_qids": sorted(hub_all & set(dev_q)),
               "w_hub_true": w_hub_true,
               "note": "recall_hiệu_chỉnh = w·recall(hub) + (1−w)·recall(không hub), w = w_hub_true"},
              f"{out_dir}/reweight.json")
    print(f"\n✅ {out_dir}/")


def main():
    ap = argparse.ArgumentParser(prog="legalir.train.split")
    ap.add_argument("--train_file", required=True, help="train.json gốc của BTC (7.000 câu)")
    ap.add_argument("--v1_dir", default="splits", help="thư mục ghi split v1")
    ap.add_argument("--v2_dir", default="splits_v2", help="thư mục ghi split v2")
    ap.add_argument("--dev_size", type=int, default=1000)
    ap.add_argument("--max_comp", type=int, default=60,
                    help="bỏ qua thành phần liên thông lớn hơn ngần này câu khi chọn dev")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stages", default="all", choices=["v1", "v2", "all"],
                    help="v2 cần dev_gt của v1 (chạy v1 trước hoặc trỏ --v1_dir có sẵn)")
    a = ap.parse_args()
    if a.stages in ("v1", "all"):
        build_v1(a.train_file, a.v1_dir, a.dev_size, a.max_comp, a.seed)
    if a.stages in ("v2", "all"):
        build_v2(a.train_file, os.path.join(a.v1_dir, "dev_gt.json"), a.v2_dir)


if __name__ == "__main__":
    main()
