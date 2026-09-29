"""Hướng 3: đào negative là văn bản CÙNG HỌ với văn bản gold (theo đồ thị trích dẫn).

Đồ thị trích dẫn đã **thất bại** khi dùng làm đặc trưng lúc inference (cộng thưởng cho
hàng xóm của top-1 làm điểm giảm đơn điệu — bậc trung vị 2 nhưng p90 là 19, đẩy lên 19
văn bản sai để cứu 1 cái đúng). Nhưng dùng để **đào negative** thì lại đúng chỗ:

  - lúc inference, đưa hàng xóm lên là canh bạc vì ta không biết cái nào đúng;
  - lúc huấn luyện, ta BIẾT cái nào đúng. Văn bản cùng họ mà KHÔNG phải gold chính là
    negative khó nhất có thể có — nó cùng chủ đề, cùng thuật ngữ, chỉ khác ở tầng/phiên
    bản. Đó đúng là kiểu nhầm số một đo được: Nghị định ↔ Nghị định 30 ca,
    Luật → Nghị định 23 ca, Thông tư → Nghị định 18 ca.

Chỉ lấy thành viên trong họ **có mặt trong pool**: vừa rẻ (đã có sẵn chunk tốt nhất của
nó), vừa đúng phân phối (đó là những cái thực sự cạnh tranh lúc inference).

Rủi ro phải canh: văn bản cùng họ RẤT dễ là false negative — nó có thể cũng trả lời được
câu hỏi mà BTC không gán nhãn. Nên chỉ loại trừ theo tập gold là chưa đủ; ta thêm một
chốt chặn: bỏ qua ứng viên mà cross-encoder chấm cao hơn cả gold, vì gần như chắc chắn
đó là positive chưa gán nhãn chứ không phải negative.
"""
from collections import Counter
import argparse
import collections
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_scores(patterns):
    out = {}
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    out.setdefault(r["qid"], {}).update(r["scores"])
    return out


def main(a):
    from eval.error_analysis import load_pool
    from eval.citation_graph import build

    pools, aid_of, _ = load_pool(a.pool_dir,
                                 {"bm25": 50, "bge": 30, "e5": 30, "bgesparse": 30})
    scores = load_scores(a.score_glob)
    gt = json.load(open(a.gt, encoding="utf-8"))
    gt = {k: (v["answer"] if isinstance(v, dict) else v) for k, v in gt.items()}
    qs = json.load(open(a.questions, encoding="utf-8"))
    qtext = {k: (v["question"] if isinstance(v, dict) else v) for k, v in qs.items()}
    _own, _edges, und = build(verbose=False)

    qids = [q for q in gt if q in scores and q in pools]

    hubs = set()
    if a.n_hub:
        # "hub" = văn bản lọt pool của nhiều câu hỏi khác nhau. Đây đúng định nghĩa
        # vận hành của nhóm gây nhiễu đã đếm ở §18 (Luật BHXH 2014, Luật Quản lý
        # thuế, Luật BHYT... đều là luật/nghị định gốc phạm vi rộng).
        freq = Counter()
        for q in qids:
            freq.update({aid_of[c] for c in pools[q]})
        cut = max(2, int(a.hub_pct * len(qids) / 100))
        hubs = {d for d, n in freq.items() if n >= cut}
        print(f"   hub: {len(hubs)} document lọt pool của ≥{cut} câu "
              f"({a.hub_pct}% của {len(qids)})")
    n_fam, n_any, n_skip, tot_fam = 0, 0, 0, 0
    drop_suspect = 0
    os.makedirs(os.path.dirname(a.out_file), exist_ok=True)
    with open(a.out_file, "w", encoding="utf-8") as out:
        for q in qids:
            best = {}
            for c in pools[q]:
                s = scores[q].get(c)
                if s is None:
                    continue
                d = aid_of[c]
                if d not in best or s > best[d][0]:
                    best[d] = (s, c)
            order = sorted(best, key=lambda d: best[d][0], reverse=True)
            gold = {str(x) for x in gt[q]}
            pos_docs = [d for d in order if d in gold]
            if not pos_docs:
                n_skip += 1
                continue
            gold_top = max(best[d][0] for d in pos_docs)

            fam = set().union(*[und.get(d, set()) for d in gold]) - gold
            fam_in_pool = [d for d in order if d in fam]
            keep = []
            for d in fam_in_pool:
                if best[d][0] > gold_top:
                    # chấm cao hơn cả gold ⇒ nhiều khả năng là positive chưa gán nhãn,
                    # không phải negative. Dạy nó là negative là dạy sai.
                    drop_suspect += 1
                    continue
                keep.append(d)
            tot_fam += len(keep)
            n_fam += bool(keep)

            negs = keep[:a.n_fam]
            if a.n_hub:
                # `result.md` §18 tự đề xuất hướng này rồi KHÔNG bao giờ chạy. Khác
                # `sem`/`clu` (đã chết) ở chỗ chúng lấy mẫu theo độ tương tự trên TOÀN
                # corpus; ở đây chỉ nhắm đúng nhóm nhỏ văn bản gốc phạm vi rộng liên tục
                # đội gold hẹp ra khỏi top-5. Vẫn áp bộ lọc "chấm cao hơn gold" như
                # nhánh cùng họ — không thì lặp lại đúng lỗi nhãn mà `guard` đã sửa.
                for d in order[:a.neg_to]:
                    if len(negs) >= a.n_fam + a.n_hub:
                        break
                    if d in hubs and d not in gold and d not in negs \
                            and best[d][0] <= gold_top:
                        negs.append(d)
            for d in order[:a.neg_to]:                  # bù bằng negative theo hạng
                if len(negs) >= a.n_neg:
                    break
                if d not in gold and d not in negs:
                    negs.append(d)
            if not negs:
                n_skip += 1
                continue
            n_any += 1
            if a.pos_all:
                # MỌI chunk của gold doc có mặt trong pool, không chỉ chunk điểm cao
                # nhất. Lúc suy luận ta lấy max qua tất cả chunk của doc; bản cũ chỉ
                # dạy đúng MỘT chunk ⇒ train và inference lệch nhau ở đúng chỗ đó.
                # Khớp với lỗi đã đo: gold càng NGẮN (ít chunk) càng dễ trượt.
                pos_chunks = [c for c in pools[q]
                              if aid_of[c] in gold and scores[q].get(c) is not None]
            else:
                pos_chunks = [best[d][1] for d in pos_docs]
            out.write(json.dumps({
                "qid": q, "query": qtext[q],
                "pos": pos_chunks,
                "neg": [best[d][1] for d in negs],
                "n_family": len(keep[:a.n_fam]),
            }, ensure_ascii=False) + "\n")

    print(f"✅ {a.out_file} · {n_any} mẫu · bỏ {n_skip} câu")
    print(f"   câu có ít nhất 1 negative CÙNG HỌ trong pool: {n_fam}/{len(qids)} "
          f"({100*n_fam/max(len(qids),1):.1f}%)")
    print(f"   trung bình {tot_fam/max(n_fam,1):.1f} negative cùng họ mỗi câu có họ")
    print(f"   loại {drop_suspect} ứng viên vì chấm CAO HƠN gold (nghi positive chưa gán nhãn)")
    if n_fam / max(len(qids), 1) < 0.15:
        print("\n⚠️  Dưới 15% số câu có negative cùng họ — cơ chế này chạm được quá ít "
              "dữ liệu để tạo khác biệt. Cân nhắc bỏ trước khi tốn 95 phút train.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pool_dir", default="eval/pools/train_fast")
    p.add_argument("--score_glob", nargs="+",
                   default=["eval/scores/train_fast/scores_eval__ckpt__vn_reranker_ft_r2__epoch2.shard*of4.jsonl.gz"])
    p.add_argument("--gt", default="eval/splits_v2/train.json")
    p.add_argument("--questions", default="eval/splits_v2/train.json")
    p.add_argument("--out_file", default="eval/train_pairs/legalir_pairs_fam.jsonl")
    p.add_argument("--n_fam", type=int, default=4, help="tối đa bao nhiêu negative cùng họ")
    p.add_argument("--n_neg", type=int, default=7)
    p.add_argument("--neg_to", type=int, default=25)
    p.add_argument("--n_hub", type=int, default=0,
                   help="số negative lấy từ danh sách hub (0 = tắt, như bản fam)")
    p.add_argument("--hub_pct", type=float, default=2.0,
                   help="doc lọt pool của ≥ ngần này %% số câu thì tính là hub")
    p.add_argument("--pos_all", action="store_true",
                   help="mọi chunk của gold doc trong pool làm positive")
    main(p.parse_args())
