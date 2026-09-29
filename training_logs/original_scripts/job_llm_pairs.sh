#!/usr/bin/env bash
# Nguyen lieu cho tang LLM: pool + diem cross-encoder cho 5.206 cau TRAIN.
#
# Phai cham cross-encoder tren pool cua tap train vi cap huan luyen cua tang LLM duoc
# dinh nghia theo THU HANG DOCUMENT ma cross-encoder dua ra — do dung la thu tang LLM
# se nhin thay luc inference. Khong co cach nao re hon: 5.206 x 96 = 500K cap.
# Chia 4 shard.
set -u
cd "$(dirname "$0")/.."
PY=/home/vannk/.venvs/uit_eval311/bin/python
LOG=eval/logs/job_llm_pairs.log
LOCK=/tmp/uit_dsc_gpu_pick.lock
CK=eval/ckpt/vn_reranker_ft_r2/epoch2
TAG=${CK//\//__}
say(){ echo "[$(date '+%m-%d %H:%M')] $*" >> $LOG; }

# FORCE_GPU o day la "UU TIEN", khong phai "ep buoc": dung GPU do CHUNG NAO no con du
# cho, het cho thi tu tran sang GPU khac. Nho vay 4 shard tu lap day GPU vua giai phong
# roi moi di noi khac, khong phai chia tay cung khong dam OOM vi nhoi qua nhieu vao mot the.
pick(){
  while :; do
    if [ -n "${FORCE_GPU:-}" ]; then
      f=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
          | awk -F', ' -v w="$FORCE_GPU" '$1==w{split($2,u," ");split($3,t," ");print t[1]-u[1]}')
      [ -n "$f" ] && [ "$f" -ge "$1" ] && { echo "$FORCE_GPU"; return; }
    fi
    g=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
        | awk -F', ' '{split($2,u," ");split($3,t," ");print t[1]-u[1], $1}' | sort -rn | head -1)
    [ "${g% *}" -ge "$1" ] && { echo "${g#* }"; return; }
    sleep 45
  done
}

# WAIT_FOR_PID: doi tien trinh do thoat roi chop ngay GPU cua no. Poll 5s chu khong 60s
# — khe trong tren may dung chung chi ton tai vai giay.
if [ -n "${WAIT_FOR_PID:-}" ]; then
  say "xep hang sau PID $WAIT_FOR_PID (uu tien GPU ${FORCE_GPU:-?})"
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do sleep 5; done
  say "PID $WAIT_FOR_PID da xong, nhay vao ngay"
fi

# --- pool cho tap train ---
while [ ! -f eval/pools/train_fast/timing_fast.json ]; do
  exec 9>"$LOCK"; flock 9
  # KHOA = NGUONG CHO, khong nho hon. Cho 6000MiB roi chi khoa 5,0GB la tu tay chua
  # 1GB cho nguoi khac chen vao. reserve_vram tu cat xuong (free - 0,4GB cho CUDA
  # context) nen xin bang ca khe khong bao gio cap phat qua tay.
  # Dinh that cua step1 5.206 cau chi ~2,5GB (sau khi chia khoi 512 cau trong
  # DenseMatrix.search_batch); nguong 6000 la dinh + dem, va ta giu tron khe do.
  G=$(pick 6000); say "step1 tap train tren GPU $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u retrieval/run_pipeline_fast.py --mode step1 \
    --question_file eval/splits_v2/train.json --save_dir eval/pools/train_fast \
    --reserve_gb 5.9 \
    --e5_model eval/ckpt/e5_ft/epoch2 --e5_dense_prefix eval/cache/e5ft_dense \
    >> eval/logs/llm_pairs_step1.log 2>&1 &
  P=$!; sleep 75; flock -u 9; exec 9>&-; wait $P
  [ -f eval/pools/train_fast/timing_fast.json ] || { say "step1 that bai, cho 4 phut"; sleep 240; }
done
say "pool train xong"

# --- cham cross-encoder, 4 shard ---
for i in 0 1 2 3; do
(
  OUT=eval/scores/train_fast/scores_${TAG}.shard${i}of4.jsonl.gz
  while [ ! -f "$OUT" ]; do
    exec 8>"$LOCK"; flock 8
    # cross-encoder batch 64 len 512: dinh ~3GB. Cho 5000MiB va giu TRON khe do.
    G=$(pick 5000); say "cham shard $i tren GPU $G"
    CUDA_VISIBLE_DEVICES=$G $PY -u eval/score_pairs.py \
      --pool_dir eval/pools/train_fast --out_dir eval/scores/train_fast \
      --reranker "$CK" --channels bm25:50 bge:30 e5:30 bgesparse:30 \
      --reserve_gb 4.9 \
      --shard $i --num_shards 4 >> eval/logs/llm_pairs_score_$i.log 2>&1 &
    P=$!; sleep 75; flock -u 8; exec 8>&-; wait $P
    [ -f "$OUT" ] || { say "shard $i that bai, cho 4 phut"; sleep 240; }
  done
  say "shard $i xong"
) &
done
wait

# --- dao cap muc document ---
$PY eval/mine_llm_pairs.py \
  --pool_dir eval/pools/train_fast \
  --score_glob "eval/scores/train_fast/scores_${TAG}.shard*of4.jsonl.gz" \
  --gt eval/splits_v2/train.json --questions eval/splits_v2/train.json \
  --out_file eval/train_pairs/llm_pairs.jsonl >> $LOG 2>&1
say "XONG — eval/train_pairs/llm_pairs.jsonl"
