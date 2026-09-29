# UIT DSC 2026 · Task 1 LegalIR — Private Test Reproduction Package (Team Artiz)

Self-contained package: MIT-licensed inference pipeline (retrieval → rerank → aggregate → submission), checkpoints, pre-built indexes, and training logs.

**Reproduced submission: `fam_twinall_nosalary.zip` — private 0.9491**, built with `legalir <private-official.json> fam`.

## 0. Result

**#1 on the public board · #12 on the private board.**

| split | submission | Recall |
|---|---|---:|
| public | `submission_cleanpool_twin070_nopool.zip` | **0,9599** |
| private | `fam_twinall_nosalary.zip` (submission id 937899) | **0,9491** |

**Model weights (Google Drive):** https://drive.google.com/file/d/1XdJujbmwTsvAoNEp4L7N2qkOrxbHYJUJ/view?pli=1
(`legalir_models.zip`, ≈8.4 GB — the 5 checkpoints of the table in §1, matching `models/` byte for
byte; verify against `../check/SHA256SUMS`).

---

## 1. Pipeline

```
question ──┬─ BM25 (CSR)                           top-50 ─┐
           ├─ bge-m3 lexical weights (sparse CSR)  top-30 ─┼─► pool ~95 chunks/question
           ├─ Vietnamese_Embedding_v2 FT (CLS)     top-30 ─┤
           └─ vietlegal-harrier-0.6b FT (last)     top-30 ─┘
                                   │
           cross-encoder Vietnamese_Reranker FT scores (question, chunk), 512 tokens
                                   │
           aggregate chunks → documents: LSE T=2 over logits → top-10 documents
                                   │
           Qwen3-Reranker-0.6B (base) scores (question, document heading), 1024 tokens
                                   │
           final score = z(LSE) + 0.15·z(heading) → cut at 5 documents
           (pool covers < 5 docs: backfill from depth-300 pool)
                                   │
           R3 twin questions: train.json neighbors with char-TFIDF cos ≥ 0.70
           → insert all gold docs of unmatched neighbors, filling slots 5, 4, 3, …
                                   │
                            submission.zip
```

- Corpus: 8,532 documents → 184,368 chunks (450 tokens, 1-sentence overlap).
- 5 checkpoints, total 2.895B parameters (< 4B rule). No generative LLM, no external API (`HF_HUB_OFFLINE=1`).

| component | checkpoint |
|---|---|
| dense channel 1 | `vnembv2_ft_ep2` (FT `AITeamVN/Vietnamese_Embedding_v2`) |
| dense channel 2 | `harrier_ft_ep2` (FT `mainguyen9/vietlegal-harrier-0.6b`) |
| sparse | `bge-m3` (base `BAAI/bge-m3`) |
| cross-encoder | `ce_fam_ep1` (FT `AITeamVN/Vietnamese_Reranker`) |
| heading reranker | `Qwen3-Reranker-0.6B` (base `Qwen/Qwen3-Reranker-0.6B`) |

## 2. Usage

### Docker

Requirements: 1 NVIDIA GPU ≥ 11 GB (peak ~5 GB), NVIDIA Container Toolkit, ~40 GB disk.

```bash
docker build -t uitdsc2026-legalir:private .

docker run --rm --gpus '"device=0"' -v /path/to/data:/data \
  -e CORPUS_DIR=/data/selected-contexts \
  uitdsc2026-legalir:private /data/private-official.json fam /data/out_fam
# -> /data/out_fam/submission.zip (exactly one submission.json)
```

The image ships two virtualenvs (`/opt/venv_retr`: transformers 5.16.1 · `/opt/venv_ce`: transformers 4.44.2 + FlagEmbedding 1.2.11). `ENTRYPOINT` is the console script `legalir` (see `pyproject.toml`).

### Without Docker

```bash
python3.11 -m venv /opt/venv_retr && /opt/venv_retr/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
  && /opt/venv_retr/bin/pip install -r env/requirements_retrieval.txt
python3.11 -m venv /opt/venv_ce   && /opt/venv_ce/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
  && /opt/venv_ce/bin/pip install -r env/requirements_ce.txt

export PY_RETR=/opt/venv_retr/bin/python PY_CE=/opt/venv_ce/bin/python
CUDA_VISIBLE_DEVICES=0 CORPUS_DIR=/path/to/selected-contexts \
  python scripts/run_inference.py /path/to/private-official.json fam
```

The pipeline is resumable: completed steps (pool, cross-encoder scores) are skipped. Per-step logs are in `output/<cfg>/logs/`.

Runtime for 2,080 questions on 1 GPU: retrieval ~5 min · cross-encoder ~60 min · headings ~16 min.

### Rebuilding indexes (if the corpus changes)

`cli.py` compares the SHA-1 of every passage against `index/corpus_manifest.json` and stops on any new/changed document. To rebuild, run inside the delivered image:

```bash
docker run --rm --gpus '"device=0"' -v <corpus>:/data:ro -v <out>:/out \
  --entrypoint /bin/bash uitdsc2026-legalir:private \
  /app/scripts/rebuild_index.sh --corpus_dir /data/selected-contexts --out_dir /out/index
```

Steps: chunk 450 tokens → BM25 → sparse bge-m3 → 2 dense matrices → headings → corpus fingerprints. Completed steps are skipped; see `rebuild_index.sh --help` for flags.

## 3. Directory layout

```
LICENSE · pyproject.toml        MIT (Team Artiz) · `legalir` package + console script
configs/fam.yaml                checkpoints + depths of the single configuration
scripts/                        inference fallback CLI, corpus/index builders, training chains
legalir/                        the package: cli · config · data · encoders · index ·
                                retrieve · rerank_ce · aggregate · headings · rerank_qwen3 ·
                                rules (R3) · blend · validate
models/                         5 checkpoints (9.8 GB)
index/                          BM25, sparse, 2 dense matrices, chunk texts, headings, fingerprints
training_logs/                  training logs (seed 42), train pairs, original scripts, TRAINING.md
env/                            pinned requirements of the 2 environments
output/                         run outputs (not packaged)
```
