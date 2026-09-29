#!/usr/bin/env bash
# Huong 3: fine-tune tiep tu vong 2 nhung DOI NGUON NEGATIVE sang van ban CUNG HO
# (hang xom trong do thi trich dan cua van ban gold) thay vi negative theo thu hang.
# So sanh cong bang: cung checkpoint xuat phat, cung lr, cung so epoch — khac duy nhat
# la nguon negative. 50,6% cau train co it nhat 1 negative cung ho (trung binh 3,2).
set -u
cd "$(dirname "$0")/.."
PY=/home/vannk/.venvs/uit_eval311/bin/python
LOG=eval/logs/job_family.log
OUT=eval/ckpt/vn_reranker_ft_fam
say(){ echo "[$(date '+%m-%d %H:%M')] $*" >> $LOG; }
. "$(dirname "$0")/gpu_lib.sh"

train(){
  say "train negative cung ho tren GPU $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u eval/train_reranker.py \
    --model eval/ckpt/vn_reranker_ft_r2/epoch2 \
    --pairs eval/train_pairs/legalir_pairs_fam.jsonl \
    --optim8bit --reserve_gb 4.4 --groups_per_batch 1 --grad_accum 8 \
    --epochs 2 --lr 3e-6 --out_dir "$OUT" >> eval/logs/train_fam.log 2>&1
}
while [ ! -d "$OUT/epoch2" ]; do
  run_on_gpu 4500 train
  [ -d "$OUT/epoch2" ] || { say "that bai, cho 5 phut"; sleep 300; }
done
say "train xong"

for EP in epoch1 epoch2; do
  RES=eval/results/devv2_fam_$EP.json
  evalone(){
    say "eval $EP tren GPU $G"
    CUDA_VISIBLE_DEVICES=$G $PY -u retrieval/run_pipeline_fast.py \
      --question_file eval/splits_v2/dev_questions.json --output_file "$RES" \
      --reranker "$OUT/$EP" \
      --e5_model eval/ckpt/e5_ft/epoch2 --e5_dense_prefix eval/cache/e5ft_dense \
      --reserve_gb 4.9 >> eval/logs/eval_fam_$EP.log 2>&1
  }
  while [ ! -f "$RES" ]; do
    run_on_gpu 5000 evalone
    [ -f "$RES" ] || { say "eval $EP that bai, cho 4 phut"; sleep 240; }
  done
  say "--- NEGATIVE CUNG HO · $EP (moc so sanh: 0,9404) ---"
  $PY eval/score_dev_v2.py "$RES" >> $LOG 2>&1
done
say "XONG"
