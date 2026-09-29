#!/usr/bin/env bash
# Ca hai kenh dieu fine-tune: bge=vnembv2_ft (thay vnembv2 pretrained) + e5=harrier_ft
# (thay e5_ft), doi voi baseline B1 = bge=vnembv2 pretrained + e5=e5_ft (dev v2 tai can
# 0.9426). Day la cau hinh da nop leaderboard public = 0.9471 (submission_vnembv2ft_harrierft.zip),
# kiem tra xem dev v2 co cho tin hieu duong tuong tu khong.
# Can venv harrier_test vi e5=harrier_ft phai qua SentenceTransformer nhan dien Qwen3.
set -u; cd "$(dirname "$0")/.."
PY=/home/vannk/.venvs/harrier_test/bin/python
ALLOW_GPUS=${ALLOW_GPUS:-"0 1 2 3 4 5 6 7"}
. eval/gpu_lib.sh
LOG=eval/logs/job_step1_vnembv2ft_harrierft.log
Q=eval/splits_v2/dev_questions.json
SAVE=eval/pools/devv2_vnembv2ft_harrierft

step1(){
  local N=$(zcat "$SAVE/chan_bm25.jsonl.gz" 2>/dev/null | wc -l)
  if [ "$N" = "1794" ]; then
    echo "[$(date '+%H:%M')] pool da du 1794 cau, bo qua" >> "$LOG"
    return
  fi
  run(){
    echo "[$(date '+%H:%M')] step1 tren GPU $G" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$G $PY -u retrieval/run_pipeline_fast.py --mode step1 \
      --question_file "$Q" --save_dir "$SAVE" \
      --index_dir retrieval/db_fast --text_source retrieval/db_fast/texts.json.gz \
      --bge_dense_model eval/ckpt/vnembv2_ft/epoch2 --bge_dense_prefix eval/cache/vnembv2ft_dense \
      --e5_model eval/ckpt/harrier_ft/epoch2 --e5_dense_prefix eval/cache/harrierft_dense --e5_no_prefix \
      --top_k_bm25 50 --top_k_dense 30 --reserve_gb 2.6 \
      >> eval/logs/step1_vnembv2ft_harrierft.log 2>&1
  }
  local t=0
  while [ "$(zcat "$SAVE/chan_bm25.jsonl.gz" 2>/dev/null | wc -l)" != "1794" ] && [ $t -lt 6 ]; do
    run_on_gpu 3200 run; t=$((t+1))
    [ "$(zcat "$SAVE/chan_bm25.jsonl.gz" 2>/dev/null | wc -l)" = "1794" ] || sleep 60
  done
  local N2=$(zcat "$SAVE/chan_bm25.jsonl.gz" 2>/dev/null | wc -l)
  [ "$N2" = "1794" ] && echo "[$(date '+%H:%M')] pool xong ($N2 cau)" >> "$LOG" \
    || echo "[$(date '+%H:%M')] pool THAT BAI ($N2 cau)" >> "$LOG"
}

rerank(){
  local OUT=$SAVE/scores_eval__ckpt__vn_reranker_ft_fam__epoch1.shard0of1.jsonl.gz
  if [ ! -f "$OUT" ]; then
    sc(){
      echo "[$(date '+%H:%M')] rerank tren GPU $G" >> "$LOG"
      CUDA_VISIBLE_DEVICES=$G $PY -u eval/score_pairs.py \
        --pool_dir "$SAVE" --out_dir "$SAVE" \
        --reranker eval/ckpt/vn_reranker_ft_fam/epoch1 \
        --channels bm25:50 bge:30 e5:30 bgesparse:30 \
        --max_length 512 --batch_size 32 --reserve_gb 3.4 >> eval/logs/rerank_vnembv2ft_harrierft.log 2>&1
    }
    local t=0
    while [ ! -f "$OUT" ] && [ $t -lt 6 ]; do run_on_gpu 3900 sc; t=$((t+1)); [ -f "$OUT" ] || sleep 60; done
  fi
  [ -f "$OUT" ] || { echo "[$(date '+%H:%M')] cham THAT BAI" >> "$LOG"; return; }
  echo "=== vnembv2ft_harrierft ===" >> "$LOG"
  $PY eval/score_from_scores.py "$OUT" --pool_dir "$SAVE" --out "$SAVE/top5_raw.json" >> "$LOG" 2>&1
}

step1
rerank
echo "XONG $(date '+%H:%M')" >> "$LOG"
tail -30 "$LOG"
