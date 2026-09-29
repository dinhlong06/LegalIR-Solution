#!/usr/bin/env bash
# Vòng r1/r2 dùng chung trước chuỗi CE fam (scripts/train_fam.sh gọi file này, stage `shared`):
#   s0 split train/dev v2 → s1 encode base (bge-m3 CLS + harrier gốc) → s2 mine r1 → s3 CE r1 →
#   s4 mine r2 (rerank bằng CE r1) → s5 train r2: vnembv2_ft + harrier_ft (retriever suy luận,
#   models/vnembv2_ft_ep2 · models/harrier_ft_ep2) + CE r2 (điểm khởi tạo của CE fam).
# Tách từ `train_cleanpool.sh` cũ (23/09, khi gói chỉ còn cấu hình fam) — các stage giữ nguyên
# từng dòng; bỏ s6–s11 (chuỗi CE cleanpool). Runner gốc đã chạy thật: training_logs/original_scripts.
#
#   STAGE=s0split|s1encode_base|s2mine_r1|s3ce_r1|s4mine_r2|s5train_r2|all \
#   CORPUS_DIR=<selected-contexts> bash scripts/train_r2.sh <train.json> [work_dir]
#
# YÊU CẦU: mạng HF (Vietnamese_Embedding_v2, vietlegal-harrier-0.6b, Vietnamese_Reranker), 1 GPU.
# CHƯA TỪNG CHẠY end-to-end trong gói — bản chép trung thực để audit; checkpoint trong models/ là chuẩn.
set -u; cd "$(dirname "$0")/.."
TRAIN_JSON=${1:?"dùng: bash scripts/train_r2.sh <train.json> [work_dir]"}
WORK=${2:-output/train_fam}
[ -f "$TRAIN_JSON" ] || { echo "⛔ thiếu train.json: $TRAIN_JSON"; exit 1; }
[ -d "${CORPUS_DIR:-}" ] || { echo "⛔ thiếu CORPUS_DIR (selected-contexts)"; exit 1; }

PY_RETR=${PY_RETR:-/opt/venv_retr/bin/python}   # transformers 5.x · ST 6 (harrier/Qwen3)
PY_CE=${PY_CE:-/opt/venv_ce/bin/python}         # transformers 4.44.2 · FlagEmbedding 1.2.11
for p in "$PY_RETR" "$PY_CE"; do [ -x "$p" ] || { echo "⛔ không thấy $p"; exit 1; }; done
# shellcheck disable=SC1091
. scripts/gpu_lib.sh
export HF_HOME="$WORK/.hf_cache"                # KHÔNG offline: cần tải 3 model gốc
unset PYTORCH_CUDA_ALLOC_CONF                   # xung đột với gpu_reserve (xem module)
export PYTHONPATH="$PWD"

VBASE=AITeamVN/Vietnamese_Embedding_v2
HBASE=mainguyen9/vietlegal-harrier-0.6b
RBASE=AITeamVN/Vietnamese_Reranker
BGE=models/bge-m3
STAGE=${STAGE:-all}
FORCE=${FORCE:-0}
TS=$(date +%Y%m%d_%H%M); LOG=$WORK/logs/train_r2_$TS.log
mkdir -p "$WORK/logs"
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }

SPLITV1=$WORK/splits; SPLITV2=$WORK/splits_v2; TV2=$WORK/splits_v2/train.json
R1P=$WORK/pairs/r1.jsonl; R2P=$WORK/pairs/r2.jsonl
BDE=$WORK/dense_r1; CE1=$WORK/ckpt/ce_r1
VFT=$WORK/ckpt/vnembv2_ft; HFT=$WORK/ckpt/harrier_ft; R2C=$WORK/ckpt/reranker_r2
need_gpu(){ run_on_gpu "$1" "$2"; }


s0_split(){
  [ -f "$TV2" ] && { say "có sẵn split ($TV2), bỏ qua"; return 0; }
  say "S0 split train/dev"
  "$PY_CE" -m legalir.train.split --train_file "$TRAIN_JSON" \
    --v1_dir "$SPLITV1" --v2_dir "$SPLITV2" 2>&1 | tee -a $LOG
  [ -f "$TV2" ] || { say "SPLIT THẤT BẠI"; return 1; }
}

s1_encode_base(){
  # Ma trận dense của model GỐC cho vòng mine r1 (bm25/sparse/texts dùng lại bản đóng
  # gói trong index/db_fast vì corpus-derived; sparse là bge-m3-derived và r1 bge
  # chính là bge-m3). Tái lập đúng thứ tự chunk nhờ chunk_ids.json đóng gói.
  for spec in "cls:$BGE:$BDE/bge_m3" "last_token:$HBASE:$BDE/harrier_base"; do
    pooling=${spec%%:*}; rest=${spec#*:}; model=${rest%%:*}; pfx=${rest##*:}
    [ -f "$pfx.npy" ] && { say "có sẵn $pfx.npy, bỏ qua"; continue; }
    say "S1 encode base $model ($pooling)"
    for i in 0 1 2 3; do
      en(){ say "encode base $(basename $pfx) shard $i · GPU $G"
        CUDA_VISIBLE_DEVICES=$G "$PY_CE" -u -m legalir.index.dense encode \
          --model "$model" --pooling "$pooling" --prefix "" --index_dir index/db_fast \
          --out_prefix "$pfx" --shard $i --num_shards 4 2>&1 | tee -a $LOG; }
      k=0; while [ ! -f "$pfx.shard${i}of4.npy" ] && [ $k -lt 8 ]; do
        need_gpu 5000 en; k=$((k+1)); done
    done
    "$PY_CE" -u -m legalir.index.dense merge --index_dir index/db_fast \
      --out_prefix "$pfx" --num_shards 4 2>&1 | tee -a $LOG
    [ -f "$pfx.npy" ] || { say "ENCODE BASE THẤT BẠI ($pfx)"; return 1; }
  done
}

s2_mine_r1(){
  [ -f "$R1P" ] && { say "có sẵn $R1P ($(wc -l < $R1P) mẫu), bỏ qua"; return 0; }
  mr(){ say "mine r1 · GPU $G"
    CUDA_VISIBLE_DEVICES=$G "$PY_CE" -u -m legalir.train.mine_pairs \
      --train_file "$TV2" --out_file "$R1P" --index_dir index/db_fast \
      --bge_model "$BGE" --e5_model "$HBASE" \
      --bge_prefix "$BDE/bge_m3" --e5_prefix "$BDE/harrier_base" \
      --texts index/db_fast/texts.json.gz --depth 50 --depth_bm25 50 \
      --neg_from 10 --neg_to 50 --n_neg 7 --seed 42 2>&1 | tee -a $LOG; }
  t=0; while [ ! -f "$R1P" ] && [ $t -lt 6 ]; do need_gpu 5000 mr; t=$((t+1)); done
  [ -f "$R1P" ] || { say "MINE R1 THẤT BẠI"; return 1; }
}

s3_ce_r1(){
  [ -d "$CE1/epoch2" ] && { say "có sẵn $CE1/epoch2, bỏ qua"; return 0; }
  tr(){ say "train CE r1 · GPU $G"
    CUDA_VISIBLE_DEVICES=$G "$PY_CE" -u -m legalir.train.train_reranker \
      --model "$RBASE" --pairs "$R1P" --texts index/db_fast/texts.json.gz \
      --loss listwise --n_neg 7 --epochs 2 --lr 1e-5 --seed 42 --optim8bit \
      --out_dir "$CE1" 2>&1 | tee -a $LOG; }
  t=0; while [ ! -d "$CE1/epoch2" ] && [ $t -lt 20 ]; do need_gpu 6200 tr; t=$((t+1)); done
  [ -d "$CE1/epoch2" ] || { say "TRAIN CE R1 THẤT BẠI"; return 1; }
}

s4_mine_r2(){
  [ -f "$R2P" ] && { say "có sẵn $R2P ($(wc -l < $R2P) mẫu), bỏ qua"; return 0; }
  mr(){ say "mine r2 (rerank CE r1) · GPU $G"
    CUDA_VISIBLE_DEVICES=$G "$PY_CE" -u -m legalir.train.mine_pairs \
      --train_file "$TV2" --out_file "$R2P" --index_dir index/db_fast \
      --bge_model "$BGE" --e5_model "$HBASE" \
      --bge_prefix "$BDE/bge_m3" --e5_prefix "$BDE/harrier_base" \
      --texts index/db_fast/texts.json.gz --rerank_model "$CE1/epoch2" \
      --depth 50 --depth_bm25 50 --neg_from 10 --neg_to 50 --n_neg 7 \
      --rerank_depth 40 --gold_topk 6 --seed 42 2>&1 | tee -a $LOG; }
  t=0; while [ ! -f "$R2P" ] && [ $t -lt 6 ]; do need_gpu 5000 mr; t=$((t+1)); done
  [ -f "$R2P" ] || { say "MINE R2 THẤT BẠI"; return 1; }
  say "kỳ vọng: trùng byte training_logs/pairs/legalir_pairs_r2.jsonl (kiểm tay)"
}

s5_train_r2(){
  # 3 model song song như stage_train gốc: vnembv2 (CLS) + harrier (last-token) + CE r2.
  # Giá trị đúng TRAINING.md: InfoNCE 3neg τ0.02 q64 d384, batch hiệu dụng 64 (2×32),
  # lr1e-5 warmup0.1 2ep, freeze embedding, 8-bit Adam, seed42; CE: listwise lr5e-6 2ep.
  for d in "$VFT" "$HFT" "$R2C"; do
    if [ -e "$d" ] && [ "$FORCE" != "1" ]; then say "từ chối train đè $d (xóa tay/FORCE=1)"; exit 1; fi
  done
  tv(){ say "train vnembv2 r2 · GPU $G"
    CUDA_VISIBLE_DEVICES=$G "$PY_CE" -u -m legalir.train.train_biencoder \
      --model "$VBASE" --pairs "$R2P" --texts index/db_fast/texts.json.gz \
      --pooling cls --q_prefix "" --d_prefix "" --max_pos 1 --n_neg 3 \
      --batch_size 2 --grad_accum 32 --epochs 2 --lr 1e-5 --warmup 0.1 --temperature 0.02 \
      --optim8bit --freeze_embeddings --reserve_gb 8.0 --seed 42 --out_dir "$VFT" 2>&1 | tee -a $LOG; }
  th(){ say "train harrier r2 · GPU $G"
    CUDA_VISIBLE_DEVICES=$G "$PY_RETR" -u -m legalir.train.train_biencoder_harrier \
      --model "$HBASE" --pairs "$R2P" --texts index/db_fast/texts.json.gz \
      --pooling last_token --q_prefix "" --d_prefix "" --max_pos 1 --n_neg 3 \
      --batch_size 2 --grad_accum 32 --epochs 2 --lr 1e-5 --warmup 0.1 --temperature 0.02 \
      --optim8bit --freeze_embeddings --reserve_gb 8.0 --seed 42 --out_dir "$HFT" 2>&1 | tee -a $LOG; }
  tr(){ say "train CE r2 · GPU $G"
    CUDA_VISIBLE_DEVICES=$G "$PY_CE" -u -m legalir.train.train_reranker \
      --model "$RBASE" --pairs "$R2P" --texts index/db_fast/texts.json.gz \
      --loss listwise --n_neg 7 --epochs 2 --lr 5e-6 --seed 42 --optim8bit \
      --out_dir "$R2C" 2>&1 | tee -a $LOG; }
  t=0; while [ ! -d "$VFT/epoch2" ] && [ $t -lt 20 ]; do need_gpu 6000 tv; t=$((t+1)); done
  t=0; while [ ! -d "$HFT/epoch2" ] && [ $t -lt 20 ]; do need_gpu 6000 th; t=$((t+1)); done
  t=0; while [ ! -d "$R2C/epoch2" ] && [ $t -lt 20 ]; do need_gpu 6200 tr; t=$((t+1)); done
  for d in "$VFT/epoch2" "$HFT/epoch2" "$R2C/epoch2"; do
    [ -d "$d" ] || { say "TRAIN R2 THIẾU $d"; return 1; }; done
  say "XONG train r2 (vnembv2 + harrier + CE)"
}


say "STAGE=$STAGE · train=$TRAIN_JSON -> $WORK"
mkdir -p "$WORK"/{pairs,ckpt,logs,splits,splits_v2,dense_r1}
case $STAGE in
  s0split) s0_split;;
  s1encode_base) s1_encode_base;;
  s2mine_r1) s2_mine_r1;;
  s3ce_r1) s3_ce_r1;;
  s4mine_r2) s4_mine_r2;;
  s5train_r2) s5_train_r2;;
  all) s0_split && s1_encode_base && s2_mine_r1 && s3_ce_r1 && s4_mine_r2 && s5_train_r2;;
  *) echo "STAGE lạ: $STAGE"; exit 1;;
esac
say "HOÀN TẤT $STAGE"
