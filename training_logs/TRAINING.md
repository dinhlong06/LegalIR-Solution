# Huấn luyện — dòng dõi checkpoint, cấu hình, log

Tài liệu phục vụ yêu cầu BTC (Điều 6): *"gửi file log huấn luyện, cấu hình môi trường để xác
minh tính tái lập trong vòng 48 giờ"*. Mọi con số dưới đây đọc thẳng từ log trong `logs/`.

## 1. Dữ liệu

Chỉ dùng dữ liệu BTC phát hành: `train.json` (7.000 câu, nhãn cấp văn bản) và
`selected-contexts` (8.532 văn bản). Không dữ liệu ngoài, không augmentation, không gán nhãn tay.
Nhãn chỉ có ở cấp **văn bản** ⇒ positive cấp chunk = chunk điểm cao nhất trong văn bản gold;
negative = chunk hạng sâu của pool retrieval (hard-negative mining trên corpus BTC).

| file cặp train (`pairs/`) | số mẫu | nguồn câu | cách đào |
|---|---:|---|---|
| `legalir_pairs_r2.jsonl` | 5.200 | train v2 (5.206 câu, tách theo văn bản khỏi dev v2) | `legalir.train.mine_pairs --rerank_model <CE vòng 1>` (vòng 2 — negative đào lại bằng CE vòng 1; log `training_logs/logs/jobs/after_r2.log`) |
| `legalir_pairs_fam.jsonl` | 5.168 | train v2 | `legalir.train.mine_family` (n_fam 4, n_neg 7, neg_to 25) trên pool `train_fast` (bm25 + **bge-m3 gốc** + **`e5_ft`** + bgesparse, 50/30/30/30), chấm bằng CE `r2/epoch2`; loại 430 ứng viên nghi positive. Runner gốc `original_scripts/job_llm_pairs.sh` (pool + chấm) — log `logs/jobs/{job_llm_pairs,llm_pairs_step1,llm_pairs_score_*}.log`. **Chạy lại bằng module trong gói trên pool + điểm gốc: trùng từng byte (23/09).** |

## 2. Dòng dõi checkpoint

```
AITeamVN/Vietnamese_Reranker ──r2 pairs, lr 5e-6, 2 ep──► vn_reranker_ft_r2/epoch2 (trung gian)
                                                              │
        ┌──── fam pairs, lr 3e-6, 2 ep (lấy ep1) ─────────────┘
        ▼
  ce_fam_ep1  [bài private 937899 = 0,9491]

intfloat/multilingual-e5-large ──r2 pairs, 2 ep──► e5_ft/epoch2 (trung gian, KHÔNG nạp lúc suy luận)
        └─► pool train_fast = bm25 + bge-m3 GỐC + e5_ft + bgesparse ──chấm CE r2/epoch2──► mine_family ──► fam pairs

AITeamVN/Vietnamese_Embedding_v2 ──r2 pairs──► vnembv2_ft_ep2
mainguyen9/vietlegal-harrier-0.6b ──r2 pairs──► harrier_ft_ep2

BAAI/bge-m3, Qwen/Qwen3-Reranker-0.6B: dùng bản gốc, không fine-tune.
```

| checkpoint trong `models/` | khởi tạo từ | cặp train | cấu hình (từ dòng `config:` của log) | log |
|---|---|---|---|---|
| `vnembv2_ft_ep2` | `AITeamVN/Vietnamese_Embedding_v2` | r2 | InfoNCE, 3 neg, τ 0,02, q_len 64, d_len 384, batch hiệu dụng 64 (2×32), lr 1e-5, warmup 0,1, 2 epoch, CLS pooling, đóng băng embedding, 8-bit Adam, seed 42 | `training_logs/logs/ckpt/vnembv2_ft/` |
| `harrier_ft_ep2` | `mainguyen9/vietlegal-harrier-0.6b` | r2 | như trên, pooling last-token | `training_logs/logs/ckpt/harrier_ft/` |
| `ce_fam_ep1` | `vn_reranker_ft_r2/epoch2` | fam | listwise (1 pos + 7 neg/nhóm), max_len 512, batch 8 nhóm (1×8), lr 3e-6, warmup 0,1, 2 epoch (**dùng epoch 1**), 8-bit Adam, seed 42 | `training_logs/logs/ckpt/vn_reranker_ft_fam/`, `logs/jobs/{job_family,train_fam}.log`, runner `original_scripts/job_family.sh` |
| (trung gian) `vn_reranker_ft_r2/epoch2` | `AITeamVN/Vietnamese_Reranker` | r2 | listwise, lr 5e-6, 2 epoch, seed 42 | `training_logs/logs/ckpt/vn_reranker_ft_r2/` |

Phần cứng: 1× NVIDIA RTX 2080 Ti 11 GB mỗi job (máy dùng chung 8 GPU), torch 2.6.0+cu124.

### 2b. Model CHỈ dùng lúc đào dữ liệu train (không nạp lúc inference)

Khai báo minh bạch cho Data Statement / Model Card. Các model dưới đây chỉ **chọn** chunk nào của corpus BTC làm
negative (và chunk nào trong văn bản gold làm positive). Chúng không sinh nhãn mềm, không distill, không sinh artefact
nào mà pipeline nộp đọc lúc suy luận. Nhãn cấp văn bản luôn là gold của BTC.

| model | gốc (danh sách BTC) | tham số | dùng để | log |
|---|---|---:|---|---|
| `vn_reranker_ft` (CE vòng 1) | `AITeamVN/Vietnamese_Reranker` | 568M | xếp lại ứng viên khi đào `legalir_pairs_r2` | `training_logs/logs/ckpt/vn_reranker_ft/` |
| `intfloat/multilingual-e5-large` (gốc) · `e5_ft` | `intfloat/multilingual-e5-large` | 560M | kênh dense của pool đào vòng 1 / vòng 2 và pool `train_fast` (cặp fam) | `training_logs/logs/ckpt/e5_ft/` |
| `vn_reranker_ft_r2/epoch2` | Vietnamese_Reranker | 568M | chấm pool `train_fast` khi đào cặp fam; đồng thời là điểm khởi tạo của CE fam | `training_logs/logs/ckpt/vn_reranker_ft_r2/` |

Ngân sách tham số của hệ thống nộp (model nạp lúc suy luận) xem README §1: **2.895.092.738**.

Đếm cả khâu huấn luyện theo **họ model** (mỗi model gốc trong danh sách BTC tính một lần; bản fine-tune không
bao giờ nạp đồng thời với bản khác cùng gốc nên quy về gốc của nó, `rule.md §2.1`): năm model của pipeline
(2.895.092.738) + `intfloat/multilingual-e5-large` chỉ dùng lúc đào dữ liệu (≈558.840.832, đếm từ `e5_ft`)
= **≈3.453.933.570 < 4.000.000.000**.

## 3. Sau khi train retriever: encode lại corpus

Vector của 184.368 chunk phải do chính retriever đã fine-tune sinh ra:
`scripts/encode_corpus.py --pooling cls` (vnembv2, `--prefix ""`) và
`scripts/encode_corpus_harrier.py --pooling last_token` (harrier), 4 shard rồi ghép, ra
`index/dense/{vnembv2ft,harrierft}_dense.{npy,ids.json}`.
Thứ tự chunk bất biến theo `bge_dense.ids.json` của index gốc.

## 4. Chạy lại chuỗi huấn luyện

### 4a. Chuỗi CE fam — `scripts/train_fam.sh`

`bash scripts/train_fam.sh <train.json> [work_dir]` (đặt `STAGE=` để chạy từng bước):
`shared` (`scripts/train_r2.sh`, s0–s5: split → mine r1 → CE r1 → mine r2 → train r2 — ra luôn hai retriever
`vnembv2_ft_ep2`, `harrier_ft_ep2` và CE r2) →
`f6train_e5` (e5_ft: `multilingual-e5-large`, cặp r2, mean pooling, tiền tố `query: `/`passage: `,
cấu hình từ log `e5_ft/train_20260812_201010.log`) → `f7encode_e5` (ma trận corpus e5_ft) →
`f8pool` (pool `train_fast`: `legalir.retrieve --bge_channel bgem3 --e5_mode e5`) → `f9score` (CE r2) →
`f10mine_family` → `f11ce_fam` (lấy **epoch1**). Mọi stage chạy bằng `venv_ce`
(FlagEmbedding 1.2.11 · sentence-transformers 2.7.0 — đúng môi trường đã dựng `train_fast`).

Ba ma trận gốc của pool `train_fast` đóng gói ở `index/train_mining/` và là mặc định của script:
`bge_sparse_orig.npz` (sha256 `caa95f5e…`), `bgem3_dense.*` (bge-m3 gốc), `e5ft_dense.*` (chỉ khớp
checkpoint e5_ft gốc — không đóng gói; đặt `E5FT_MODEL=<e5_ft/epoch2 gốc>` để dùng, không thì f6/f7
train + encode lại). Hash trong `../check/SHA256SUMS`.

**Đã kiểm 23/09 (trên máy phát triển):**
- `legalir.retrieve` chế độ đào cặp trên 5.206 câu train v2 so với pool gốc `train_fast`: **cả 4 kênh
  5.206/5.206 trùng thứ tự, max|Δ điểm| = 0** — khi dùng ba ma trận gốc (bge-m3 dense, e5ft dense,
  `bge_sparse` sha256 `caa95f5e…`). Đặt `BGE_DENSE`, `E5_DENSE`, `SPARSE_NPZ` trỏ tới chúng để tái lập bit.
  Với `index/db_fast/bge_sparse.npz` của gói (bản encode lại 18/09, sha256 `664ba…`) kênh bgesparse
  chỉ trùng tập 4.437/5.206 câu. Chạy bằng `venv_retr` (FlagEmbedding 1.4.2) thì kênh bge lệch
  (4.621/5.206) — phải dùng `venv_ce`.
- `legalir.train.mine_family` trên pool gốc + điểm CE r2 gốc: **trùng từng byte**
  `pairs/legalir_pairs_fam.jsonl` (5.168 mẫu).

### 4b. Bản gốc các script đã chạy thật

`training_logs/original_scripts/` giữ **nguyên văn** script và runner đã chạy thật lúc huấn
luyện (`train_biencoder.py`, `train_biencoder_harrier.py`, `train_reranker.py`,
`mine_training_pairs.py`, `mine_family_negatives.py`, `encode_corpus*.py`, `job_llm_pairs.sh`,
`job_family.sh`, `job_public_deep_prod.sh`, `job_step1_vnembv2ft_harrierft.sh`, `index_build/`,
`retrieval_train_fast/`). Chúng giữ đường dẫn của repo phát triển (`eval/ckpt/...`,
`/home/vannk/.venvs/...`) — là bản ghi trung thực, không phải lệnh chạy thẳng trong thư mục này.

Khác biệt giữa bản gốc và module trong `legalir/train/`:
- bỏ các nhánh thí nghiệm không dùng ở production (`--loss lsepair/doclse/bce`, `--max_pos>1`,
  LoRA, DDP nhiều GPU, `--resume_from`, nhóm cấp văn bản, mặt nạ known-gold);
- `legalir.train.train_biencoder` ghi kèm cấu hình pooling cho SentenceTransformer ngay lúc
  lưu checkpoint (`legalir/train/st_config.py`) — vá tận gốc lỗi §65: checkpoint thiếu
  `modules.json` khiến câu hỏi bị mean-pooling trong khi corpus là last-token, lệch im lặng và
  đã làm hỏng một bài nộp public;
- `legalir.train.mine_family` đã chạy lại và cho ra **đúng từng byte** `legalir_pairs_fam.jsonl` (mục 4a).
