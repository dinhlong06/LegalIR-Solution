#!/usr/bin/env bash
# Dung lai deep pool CONG KHAI bang DUNG retriever production (vnembv2_ft + harrier_ft),
# depth 300 ca 4 kenh. Bat buoc: pool `public_full_depthcheck` co san (31/08) chi nest
# 89,7% o kenh e5 va 96,3% o bge => no duoc dung bang model dense KHAC. Dung no lam
# nguon chunk them vao se lap lai dung bay da giet ban doc-prior: nguon dac trung doi
# giua ban do (dev) va luc nop (public).
set -u; cd "$(dirname "$0")/.."
PYH=/home/vannk/.venvs/harrier_test/bin/python
ALLOW_GPUS=${ALLOW_GPUS:-"0 1 2 3 4 5 6 7"}
. eval/gpu_lib.sh
LOG=eval/logs/job_public_deep_prod.log
SAVE=eval/pools/public_deep_prod
Q=data/LegalIR_Public_Test/public-official.json
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
n_of(){ zcat "$SAVE/chan_bm25.jsonl.gz" 2>/dev/null | wc -l; }
step1(){
  say "step1 deep pool cong khai (production retriever, depth 300) tren GPU $G"
  CUDA_VISIBLE_DEVICES=$G $PYH -u retrieval/run_pipeline_fast.py --mode step1 \
    --question_file "$Q" --save_dir "$SAVE" \
    --index_dir retrieval/db_fast --text_source retrieval/db_fast/texts.json.gz \
    --bge_dense_model eval/ckpt/vnembv2_ft/epoch2 --bge_dense_prefix eval/cache/vnembv2ft_dense \
    --e5_model eval/ckpt/harrier_ft/epoch2 --e5_dense_prefix eval/cache/harrierft_dense --e5_no_prefix \
    --top_k_bm25 300 --top_k_dense 300 --reserve_gb 2.6 \
    >> eval/logs/step1_public_deep_prod.log 2>&1
}
t=0
while [ "$(n_of)" != "1000" ] && [ $t -lt 10 ]; do
  run_on_gpu 3200 step1; t=$((t+1)); [ "$(n_of)" = "1000" ] || { say "lan $t: $(n_of) cau"; sleep 90; }
done
[ "$(n_of)" = "1000" ] && say "XONG deep pool cong khai: $(n_of) cau" || say "THAT BAI"
