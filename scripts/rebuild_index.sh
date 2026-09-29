#!/usr/bin/env bash
# Bước dựng index 2..6/6 — dựng lại TOÀN BỘ index từ corpus BTC (bước 1 là
# scripts/build_corpus_chunks.py, gọi luôn ở đây).
#
#   bash scripts/rebuild_index.sh --corpus_dir <selected-contexts> [--out_dir index]
#
# ⚠️ CHẠY TRONG IMAGE BÀN GIAO (môi trường pin) để kết quả khớp artifact đang đóng gói:
#     docker run --rm --gpus '"device=0"' -v <corpus>:/data:ro -v <out>:/out \
#       --entrypoint /bin/bash uitdsc2026-legalir:private \
#       /app/scripts/rebuild_index.sh --corpus_dir /data/selected-contexts --out_dir /out/index
#   Chạy ngoài image (venv khác / GPU khác đời) có thể lệch ở bước sparse (FlagEmbedding)
#   và dense (fp16 trên GPU). Sau khi dựng xong PHẢI chạy:
#     python check/verify_index_rebuild.py --rebuilt <out_dir> --reference ../legalir/index
#   và hồi quy public 1000/1000 (check/verify_public.py) trước khi dùng.
#
#   Cấu hình encode tái lập (đo 18/09/2026):
#     dense  : --shard i/4, batch 32, sort theo độ dài  => TRÙNG KHÍT BIT (max|Δ| = 0.0)
#     sparse : liền-mạch (1 shard), batch 8 (Kaggle 08/08) => KHÔNG bit-reproduce;
#              nhưng trích lại từ Qdrant gốc thì BYTE-IDENTICAL (đã chứng minh, xem README §6).
#
# Biến môi trường:
#   PY_RETR  python venv retrieval (transformers 5.x)   [/opt/venv_retr/bin/python]
#   PY_CE    python venv cross-encoder (FlagEmbedding)  [/opt/venv_ce/bin/python]
#   CUDA_VISIBLE_DEVICES  một GPU >= 8 GB
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Cấu hình encode đúng như lần dựng GỐC (đo 18/09/2026):
#   dense  : server, eval/encode_corpus*.py  -> --shard i/4, batch 32, sort theo độ dài
#            => tái lập TRÙNG KHÍT BIT (max|Δ| = 0.0)
#   sparse : Kaggle 08/08/2026, build_db_corpus_dsc.py cell 16 -> liền-mạch (1 shard), batch 8
#            => KHÔNG bit-reproduce (môi trường Kaggle đã mất); nhưng trích lại từ Qdrant gốc
#               thì BYTE-IDENTICAL (đã chứng minh). Xem README §6.
CORPUS=""; OUT="index"; FORCE=0; SKIP_SPARSE=0; SKIP_DENSE=0
DENSE_SHARDS=4; DENSE_BATCH=32; SPARSE_SHARDS=1; SPARSE_BATCH=8
while [ $# -gt 0 ]; do
  case "$1" in
    --corpus_dir)     CORPUS=$2; shift 2;;
    --out_dir)        OUT=$2; shift 2;;
    --dense_shards)   DENSE_SHARDS=$2; shift 2;;
    --dense_batch)    DENSE_BATCH=$2; shift 2;;
    --sparse_shards)  SPARSE_SHARDS=$2; shift 2;;
    --sparse_batch)   SPARSE_BATCH=$2; shift 2;;
    --skip_sparse) SKIP_SPARSE=1; shift;;
    --skip_dense)  SKIP_DENSE=1; shift;;
    --force)       FORCE=1; shift;;
    -h|--help)    sed -n '2,22p' "$0"; exit 0;;
    *) echo "⛔ tham số lạ: $1"; exit 1;;
  esac
done
[ -n "$CORPUS" ] || { echo "⛔ thiếu --corpus_dir"; exit 1; }
[ -d "$CORPUS" ] || { echo "⛔ không thấy corpus: $CORPUS"; exit 1; }

PY_RETR=${PY_RETR:-/opt/venv_retr/bin/python}
PY_CE=${PY_CE:-/opt/venv_ce/bin/python}
[ -x "$PY_RETR" ] || { echo "⛔ không thấy PY_RETR=$PY_RETR"; exit 1; }
[ -x "$PY_CE" ]   || { echo "⛔ không thấy PY_CE=$PY_CE"; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

OUT=$([ "${OUT:0:1}" = "/" ] && echo "$OUT" || echo "$ROOT/$OUT")
DB=$OUT/db_fast; DEN=$OUT/dense
mkdir -p "$DB" "$DEN" "$OUT/logs"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$OUT/logs/rebuild.log"; }
have(){ [ -f "$1" ] && [ "$FORCE" = 0 ]; }

# vân tay môi trường — để audit yêu cầu 48h của BTC (pip freeze 2 venv + torch/cuda/driver/GPU)
{
  echo "# env_fingerprint — $(date '+%F %T')"
  echo "## uname"; uname -a
  echo "## nvidia-smi (driver + GPU)"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "n/a"
  for v in PY_RETR PY_CE; do
    p=${!v}
    echo "## $v = $p"
    "$p" -c "import sys, torch; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>/dev/null || true
    "$p" -m pip freeze 2>/dev/null || true
  done
} > "$OUT/env_fingerprint.txt"
say "=== rebuild index · corpus=$CORPUS -> $OUT · shards=$SHARDS ==="

# [1/6] chunk + universe -------------------------------------------------------
if have "$DB/chunk_ids.json" && have "$DB/texts.json.gz"; then
  say "[1/6] chunk đã có — bỏ qua (dùng --force để dựng lại)"
else
  say "[1/6] chunk corpus -> texts.json.gz + chunk_ids.json"
  "$PY_CE" scripts/build_corpus_chunks.py --corpus_dir "$CORPUS" --out_dir "$DB" \
    2>&1 | tee -a "$OUT/logs/rebuild.log"
fi

# [2/6] BM25 CSR ---------------------------------------------------------------
if have "$DB/bm25_W.npz"; then say "[2/6] BM25 đã có — bỏ qua"
else
  say "[2/6] BM25 (rank-bm25, float64)"
  "$PY_CE" -u -m legalir.index.bm25 --index_dir "$DB" 2>&1 | tee -a "$OUT/logs/step2_bm25.log"
fi

# [3/6] sparse bge-m3 (BẮT BUỘC venv CE: FlagEmbedding 1.2.11) ------------------
if [ "$SKIP_SPARSE" = 1 ]; then say "[3/6] sparse: bỏ qua theo yêu cầu"
elif have "$DB/bge_sparse.npz"; then say "[3/6] sparse đã có — bỏ qua"
else
  say "[3/6] sparse bge-m3 lexical weights (venv CE, FlagEmbedding 1.2.11; batch $SPARSE_BATCH)"
  for i in $(seq 0 $((SPARSE_SHARDS-1))); do
    "$PY_CE" -u -m legalir.index.sparse encode --index_dir "$DB" \
      --out "$DB/bge_sparse.npz" --shard "$i" --num_shards "$SPARSE_SHARDS" \
      --batch_size "$SPARSE_BATCH" 2>&1 | tee -a "$OUT/logs/step3_sparse.log"
  done
  "$PY_CE" -u -m legalir.index.sparse merge --index_dir "$DB" \
    --out "$DB/bge_sparse.npz" --num_shards "$SPARSE_SHARDS" 2>&1 | tee -a "$OUT/logs/step3_sparse.log"
fi

# [4/6] ma trận dense (pooling phải khớp corpus) --------------------------------
#   vnembv2 FT  -> CLS         (venv CE,  y hệt eval/encode_corpus.py lúc đóng gói)
#   harrier FT  -> last-token  (venv RETR, y hệt eval/encode_corpus_harrier.py)
if [ "$SKIP_DENSE" = 1 ]; then say "[4/6] dense: bỏ qua theo yêu cầu"
else
  say "[4/6] ma trận dense (vnembv2_ft_ep2 + harrier_ft_ep2 của cấu hình fam)"
  encode_dense(){ # <venv> <model> <pooling> <out_prefix>
    local py=$1 model=$2 pooling=$3 pfx=$4
    if have "$pfx.npy"; then say "  dense $(basename "$pfx") đã có — bỏ qua"; return; fi
    say "  dense $model ($pooling, shard $DENSE_SHARDS, batch $DENSE_BATCH) -> $(basename "$pfx")"
    for i in $(seq 0 $((DENSE_SHARDS-1))); do
      "$py" -u -m legalir.index.dense encode --model "$model" --pooling "$pooling" \
        --prefix "" --index_dir "$DB" --out_prefix "$pfx" --shard "$i" --num_shards "$DENSE_SHARDS" \
        --batch_size "$DENSE_BATCH" 2>&1 | tee -a "$OUT/logs/step4_dense.log"
    done
    "$py" -u -m legalir.index.dense merge --index_dir "$DB" --out_prefix "$pfx" \
      --num_shards "$DENSE_SHARDS" 2>&1 | tee -a "$OUT/logs/step4_dense.log"
  }
  encode_dense "$PY_CE"   models/vnembv2_ft_ep2  cls        "$DEN/vnembv2ft_dense"
  encode_dense "$PY_RETR" models/harrier_ft_ep2  last_token "$DEN/harrierft_dense"
fi

# [5/6] đề mục v3-strict --------------------------------------------------------
if have "$OUT/headings_v3_strict.jsonl"; then say "[5/6] headings đã có — bỏ qua"
else
  say "[5/6] đề mục v3-strict"
  "$PY_CE" scripts/build_heading_corpus_v3.py --corpus_dir "$CORPUS" \
    --out "$OUT/headings_v3_strict.jsonl" 2>&1 | tee -a "$OUT/logs/step5_headings.log"
fi

# [6/6] vân tay corpus + doc_ids -------------------------------------------------
if have "$OUT/corpus_manifest.json"; then say "[6/6] manifest đã có — bỏ qua"
else
  say "[6/6] corpus_manifest + doc_ids"
  "$PY_CE" -u -m legalir.data manifest --corpus_dir "$CORPUS" --build \
    --manifest "$OUT/corpus_manifest.json" --doc_ids "$OUT/doc_ids.json" \
    2>&1 | tee -a "$OUT/logs/step6_manifest.log"
fi

say "✅ XONG: $OUT  — giờ chạy check/verify_index_rebuild.py + hồi quy public trước khi dùng"