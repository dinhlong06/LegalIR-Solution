#!/usr/bin/env bash
# Chuỗi huấn luyện CE fam — checkpoint `models/ce_fam_ep1` của bài private 937899 (0,9491).
# Mirror các job gốc đã chạy thật (training_logs/original_scripts/):
#   job_llm_pairs.sh  (13/08) pool train `train_fast` = bm25 + bge-m3 GỐC + e5_ft + bgesparse, chấm CE r2
#   job_family.sh     (13/08) train CE fam từ r2/epoch2 trên cặp negative cùng họ
# Cặp train: mine_family_negatives.py mặc định (n_fam 4 · n_neg 7 · neg_to 25).
#
#   STAGE=shared|f6train_e5|f7encode_e5|f8pool|f9score|f10mine_family|f11ce_fam|all \
#   CORPUS_DIR=<selected-contexts> bash scripts/train_fam.sh <train.json> [work_dir]
#
# `shared` = scripts/train_r2.sh (split → encode base → mine r1 → CE r1 →
# mine r2 → train r2) chạy trên CÙNG work_dir; fam chỉ cần từ đó `pairs/r2.jsonl` và
# `ckpt/reranker_r2/epoch2`.
#
# Môi trường: mọi stage đều chạy bằng PY_CE (transformers 4.44.2 · FlagEmbedding 1.2.11 ·
# sentence-transformers 2.7.0) — ĐÚNG môi trường đã dựng train_fast. Đo 23/09: chạy bằng
# PY_RETR (FlagEmbedding 1.4.2) thì kênh bge lệch 585/5.206 câu.
#
# TÁI LẬP TỪNG BIT (đo 23/09, `legalir.retrieve` trên 5.206 câu train v2 vs eval/pools/train_fast):
#   bm25 · bge · e5 · bgesparse đều 5.206/5.206 trùng thứ tự, max|Δ điểm| = 0 với ba ma trận gốc,
#   đóng gói sẵn ở index/train_mining/ (mặc định của script này):
#     bge_sparse_orig.npz  bge-m3 sparse, sha256 caa95f5e… (bản trích Qdrant 12/08 — KHÁC
#                          index/db_fast/bge_sparse.npz của đường suy luận, bản encode lại 18/09)
#     bgem3_dense.*        bge-m3 GỐC dense (khớp models/bge-m3)
#     e5ft_dense.*         e5_ft/epoch2 dense — CHỈ khớp đúng checkpoint e5_ft gốc
#   e5_ft gốc không đóng gói: đặt E5FT_MODEL=<e5_ft/epoch2 gốc> thì bỏ qua f6/f7 và dùng
#   e5ft_dense đóng gói ⇒ pool train_fast trùng bit. Không đặt thì train e5_ft mới (f6) và encode
#   lại (f7) — pool gần giống, không cam kết bit. `mine_family` trên pool gốc + điểm CE r2 gốc cho ra
#   trùng từng byte `training_logs/pairs/legalir_pairs_fam.jsonl` (đo 23/09).
#
# YÊU CẦU: mạng HF để tải `intfloat/multilingual-e5-large` (+ 3 model gốc của stage shared).
# CHƯA TỪNG CHẠY end-to-end trong gói này — driver là bản chép trung thực để audit;
# checkpoint đã ship trong models/ mới là chuẩn.
set -u; cd "$(dirname "$0")/.."
TRAIN_JSON=${1:?"dùng: bash scripts/train_fam.sh <train.json> [work_dir]"}
WORK=${2:-output/train_fam}
PY_RETR=${PY_RETR:-/opt/venv_retr/bin/python}
PY_CE=${PY_CE:-/opt/venv_ce/bin/python}
[ -x "$PY_CE" ] || { echo "⛔ không thấy $PY_CE"; exit 1; }
# shellcheck disable=SC1091
. scripts/gpu_lib.sh
export HF_HOME="$WORK/.hf_cache"
unset PYTORCH_CUDA_ALLOC_CONF
export PYTHONPATH="$PWD"

E5BASE=intfloat/multilingual-e5-large
STAGE=${STAGE:-all}
TS=$(date +%Y%m%d_%H%M); LOG=$WORK/logs/train_fam_$TS.log
mkdir -p "$WORK"/{logs,ckpt,pairs,dense_fam,pool_train_fast,scores_train_fast}
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }

TV2=$WORK/splits_v2/train.json
R2P=$WORK/pairs/r2.jsonl
R2C=$WORK/ckpt/reranker_r2
MINE=index/train_mining
if [ -n "${E5FT_MODEL:-}" ]; then                      # e5_ft gốc ⇒ dùng ma trận đóng gói
  E5_EP2=$E5FT_MODEL; E5_DENSE=${E5_DENSE:-$MINE/e5ft_dense}
else                                                   # train lại (f6) + encode lại (f7)
  E5_EP2=$WORK/ckpt/e5_ft/epoch2; E5_DENSE=${E5_DENSE:-$WORK/dense_fam/e5ft}
fi
E5FT=$WORK/ckpt/e5_ft
BGE_DENSE=${BGE_DENSE:-$MINE/bgem3_dense}              # bge-m3 gốc dense (bản gốc 12/08)
SPARSE_NPZ=${SPARSE_NPZ:-$MINE/bge_sparse_orig.npz}    # bge-m3 sparse gốc caa95f5e…
POOL=$WORK/pool_train_fast
SCOR=$WORK/scores_train_fast
FAMP=$WORK/pairs/legalir_pairs_fam.jsonl
CEF=$WORK/ckpt/ce_fam

shared(){
  for s in s0split s1encode_base s2mine_r1 s3ce_r1 s4mine_r2 s5train_r2; do
    STAGE=$s bash scripts/train_r2.sh "$TRAIN_JSON" "$WORK" || return 1
  done
}

f6_train_e5(){
  # Cấu hình từ dòng `config:` của training_logs/logs/ckpt/e5_ft/train_20260812_201010.log:
  # InfoNCE 3 neg, τ 0,02, q_len 64, d_len 384, batch hiệu dụng 64, lr 1e-5, warmup 0,1,
  # 2 epoch, đóng băng embedding, 8-bit Adam, seed 42, tiền tố "query: "/"passage: ", mean pooling.
  [ -n "${E5FT_MODEL:-}" ] && { say "dùng e5_ft gốc $E5FT_MODEL — bỏ qua train"; return 0; }
  [ -d "$E5FT/epoch2" ] && { say "có sẵn $E5FT/epoch2, bỏ qua"; return 0; }
  te(){ say "train e5_ft (multilingual-e5-large, cặp r2) · GPU $G"
    CUDA_VISIBLE_DEVICES=$G "$PY_CE" -u -m legalir.train.train_biencoder \
      --model "$E5BASE" --pairs "$R2P" --texts index/db_fast/texts.json.gz \
      --pooling mean --q_prefix "query: " --d_prefix "passage: " --max_pos 1 --n_neg 3 \
      --q_len 64 --d_len 384 --batch_size 2 --grad_accum 32 --epochs 2 --lr 1e-5 --warmup 0.1 \
      --temperature 0.02 --optim8bit --freeze_embeddings --seed 42 --out_dir "$E5FT" 2>&1 | tee -a $LOG; }
  t=0; while [ ! -d "$E5FT/epoch2" ] && [ $t -lt 20 ]; do run_on_gpu 6000 te; t=$((t+1)); done
  [ -d "$E5FT/epoch2" ] || { say "TRAIN e5_ft THẤT BẠI"; return 1; }
}

f7_encode_e5(){
  [ -f "$E5_DENSE.npy" ] && { say "có sẵn $E5_DENSE.npy, bỏ qua"; return 0; }
  for i in 0 1 2 3; do
    en(){ say "encode e5_ft corpus shard $i · GPU $G"
      CUDA_VISIBLE_DEVICES=$G "$PY_CE" -u -m legalir.index.dense encode --model "$E5_EP2" \
        --pooling mean --prefix "passage: " --index_dir index/db_fast \
        --out_prefix "$E5_DENSE" --shard $i --num_shards 4 2>&1 | tee -a $LOG; }
    k=0; while [ ! -f "$E5_DENSE.shard${i}of4.npy" ] && [ $k -lt 8 ]; do run_on_gpu 4500 en; k=$((k+1)); done
  done
  "$PY_CE" -u -m legalir.index.dense merge --index_dir index/db_fast \
    --out_prefix "$E5_DENSE" --num_shards 4 2>&1 | tee -a $LOG
  [ -f "$E5_DENSE.npy" ] || { say "ENCODE e5_ft THẤT BẠI"; return 1; }
}

f8_pool(){
  n_of(){ zcat "$POOL/chan_bm25.jsonl.gz" 2>/dev/null | wc -l; }
  [ "$(n_of)" = "5206" ] && { say "có sẵn pool train_fast ($POOL), bỏ qua"; return 0; }
  s1(){ say "pool train_fast (bge-m3 gốc + e5_ft) · GPU $G"
    CUDA_VISIBLE_DEVICES=$G "$PY_CE" -u -m legalir.retrieve \
      --questions "$TV2" --out_dir "$POOL" --index_dir index/db_fast --sparse_npz "$SPARSE_NPZ" \
      --bge_model models/bge-m3 --bge_channel bgem3 \
      --dense_model models/bge-m3 --dense_prefix "$BGE_DENSE" \
      --e5_model "$E5_EP2" --e5_mode e5 --e5_prefix "$E5_DENSE" \
      --top_k_bm25 50 --top_k_dense 30 2>&1 | tee -a $LOG; }
  t=0; while [ "$(n_of)" != "5206" ] && [ $t -lt 8 ]; do run_on_gpu 4500 s1; t=$((t+1)); done
  [ "$(n_of)" = "5206" ] || { say "POOL THẤT BẠI"; return 1; }
}

f9_score(){
  for i in 0 1 2 3; do
    O=$SCOR/scores_ce.shard${i}of4.jsonl.gz
    [ -f "$O" ] && { say "có sẵn $O, bỏ qua"; continue; }
    sc(){ say "chấm pool train_fast shard $i (CE r2) · GPU $G"
      CUDA_VISIBLE_DEVICES=$G "$PY_CE" -u -m legalir.rerank_ce --pool "$POOL" --model "$R2C/epoch2" \
        --out "$O" --channels bm25:50 bge:30 e5:30 bgesparse:30 --max_length 512 --batch_size 32 \
        --shard $i --num_shards 4 2>&1 | tee -a $LOG; }
    t=0; while [ ! -f "$O" ] && [ $t -lt 8 ]; do run_on_gpu 4200 sc; t=$((t+1)); done
    [ -f "$O" ] || { say "shard $i THẤT BẠI"; return 1; }
  done
}

f10_mine_family(){
  [ -f "$FAMP" ] && { say "có sẵn $FAMP, bỏ qua"; return 0; }
  "$PY_CE" -m legalir.train.mine_family --pool_dir "$POOL" \
    --score_glob "$SCOR/scores_ce.shard*of4.jsonl.gz" --gt "$TV2" --questions "$TV2" \
    --graph index/citation_graph.json --out_file "$FAMP" --n_fam 4 --n_neg 7 --neg_to 25 2>&1 | tee -a $LOG
  [ -f "$FAMP" ] || { say "MINE FAMILY THẤT BẠI"; return 1; }
  cmp -s "$FAMP" training_logs/pairs/legalir_pairs_fam.jsonl \
    && say "✅ trùng byte training_logs/pairs/legalir_pairs_fam.jsonl" \
    || say "⚠️  khác training_logs/pairs/legalir_pairs_fam.jsonl (xem ghi chú TÁI LẬP TỪNG BIT ở đầu file)"
}

f11_ce_fam(){
  # job_family.sh: từ r2/epoch2, listwise 1 pos + 7 neg, 1 nhóm × accum 8, lr 3e-6, 2 epoch,
  # max_len 512, 8-bit Adam, seed 42 → production dùng EPOCH 1 (models/ce_fam_ep1).
  [ -d "$CEF/epoch2" ] && { say "có sẵn $CEF/epoch2, bỏ qua"; return 0; }
  tr(){ say "train CE fam (init r2/epoch2) · GPU $G"
    CUDA_VISIBLE_DEVICES=$G "$PY_CE" -u -m legalir.train.train_reranker \
      --model "$R2C/epoch2" --pairs "$FAMP" --texts index/db_fast/texts.json.gz \
      --loss listwise --n_neg 7 --groups_per_batch 1 --grad_accum 8 --max_length 512 \
      --epochs 2 --lr 3e-6 --seed 42 --optim8bit --reserve_gb 1.5 \
      --out_dir "$CEF" 2>&1 | tee -a $LOG; }
  t=0; while [ ! -d "$CEF/epoch2" ] && [ $t -lt 40 ]; do run_on_gpu 6200 tr; t=$((t+1)); done
  [ -d "$CEF/epoch1" ] || { say "TRAIN CE fam THẤT BẠI"; return 1; }
  say "✅ XONG chuỗi fam → $CEF/epoch1 (tương ứng models/ce_fam_ep1)"
}

say "STAGE=$STAGE · train=$TRAIN_JSON -> $WORK · sparse=$SPARSE_NPZ · bge=$BGE_DENSE · e5=$E5_EP2 ($E5_DENSE)"
case $STAGE in
  shared) shared;;
  f6train_e5) f6_train_e5;;
  f7encode_e5) f7_encode_e5;;
  f8pool) f8_pool;;
  f9score) f9_score;;
  f10mine_family) f10_mine_family;;
  f11ce_fam) f11_ce_fam;;
  all) shared && f6_train_e5 && f7_encode_e5 && f8_pool && f9_score && f10_mine_family && f11_ce_fam;;
  *) echo "STAGE lạ: $STAGE"; exit 1;;
esac
say "HOÀN TẤT $STAGE"
