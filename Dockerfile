# LegalIR (UIT DSC 2026 · Task 1) — image tái lập kết quả private test. Đội Artiz.
#
#   docker build -t uitdsc2026-legalir:private .          # chạy trong thư mục này (gói legalir/)
#   docker run --rm --gpus '"device=0"' -v /duong/dan/data:/data \
#     -e CORPUS_DIR=/data/selected-contexts \
#     uitdsc2026-legalir:private /data/private-official.json fam /data/out_fam
#
# Hai môi trường Python tách rời, đúng như lúc tạo bài nộp (không hợp nhất được):
#   /opt/venv_retr  transformers 5.16.1 · sentence-transformers 6.0.0 · FlagEmbedding 1.4.2
#                   -> bước 1 retrieval (harrier Qwen3 last-token) + bước 3 Qwen3-Reranker
#   /opt/venv_ce    transformers 4.44.2 · FlagEmbedding 1.2.11
#                   -> bước 2 cross-encoder XLM-R (FlagReranker fp16) + bước 4 luật R3 + bước 5 kiểm định dạng
# torch 2.6.0+cu124 dùng chung từ base image (--system-site-packages).
# Console script `legalir` cài editable vào venv_retr -> ENTRYPOINT.

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS env
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY env/requirements_retrieval.txt env/requirements_ce.txt /tmp/env/
RUN python -m venv --system-site-packages /opt/venv_retr \
 && /opt/venv_retr/bin/pip install -r /tmp/env/requirements_retrieval.txt \
 && python -m venv --system-site-packages /opt/venv_ce \
 && /opt/venv_ce/bin/pip install -r /tmp/env/requirements_ce.txt

# Code (MIT) — nhỏ, đặt trước trọng số để sửa code không phải chép lại 19 GB.
COPY LICENSE README.md pyproject.toml /app/
COPY legalir/ /app/legalir/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY env/ /app/env/
# Log huấn luyện + cặp train + script gốc đã chạy (đáp ứng yêu cầu log 48h của Điều 6).
COPY training_logs/ /app/training_logs/
RUN /opt/venv_retr/bin/pip install --no-deps -e .

FROM env AS full
# Trọng số + index nằm TRONG image: lúc inference không tải gì từ mạng.
COPY index/ /app/index/
COPY models/ /app/models/
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PY_RETR=/opt/venv_retr/bin/python PY_CE=/opt/venv_ce/bin/python

ENTRYPOINT ["/opt/venv_retr/bin/legalir"]
