"""Encode câu hỏi cho ba kênh cần model (bgesparse · bge · e5).

Ba hàm tách ra từ `legalir.retrieve` — giữ nguyên từng dòng để tái lập từng bit:
  - `encode_queries_bgem3` : bge-m3 GỐC, chỉ phần lexical weights được dùng ở kênh sparse;
  - `encode_queries_cls`   : bi-encoder FT họ XLM-R (vnembv2), CLS + chuẩn hoá, q_len 64;
  - `encode_queries_st`    : SentenceTransformer (harrier, nền Qwen3), last-token theo
    modules.json, KHÔNG tiền tố.
  - `encode_queries_e5`    : họ e5 (e5_ft) — CHỈ dùng khi dựng pool đào cặp CE fam.

Import nặng (torch/transformers/FlagEmbedding) nằm TRONG hàm để module này import được
ở mọi môi trường không cần GPU.
"""
import os

import numpy as np


def encode_queries_bgem3(model_path, questions, batch_size=64):
    """bge-m3 GỐC: trả (dense, chỉ số sparse, trọng số sparse) của từng câu hỏi.

    Kênh `bgesparse` chỉ dùng phần sparse; phần dense chỉ dùng khi ĐÀO cặp huấn luyện.
    Bản fine-tune không có head sparse nên kênh này luôn chạy bằng model gốc."""
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(model_path, use_fp16=True)
    dense, idx, val = [], [], []
    for i in range(0, len(questions), batch_size):
        emb = model.encode(questions[i:i + batch_size], return_dense=True,
                           return_sparse=True, return_colbert_vecs=False)
        dense.extend(emb["dense_vecs"])
        for w in emb["lexical_weights"]:
            idx.append(list(w.keys()))
            val.append(list(w.values()))
    del model
    return dense, idx, val


def encode_queries_cls(model_path, questions, q_len=64, batch_size=64):
    """Kênh `bge`: bi-encoder đã FT, CLS + chuẩn hoá, ĐÚNG như lúc encode corpus.

    Sai một trong ba thứ (pooling / chuẩn hoá / tiền tố) là lệch hệ toạ độ."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, add_pooling_layer=False)
    model = model.half().to("cuda" if torch.cuda.is_available() else "cpu").eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(questions), batch_size):
            enc = tok(questions[i:i + batch_size], padding=True, truncation=True,
                      max_length=q_len, return_tensors="pt").to(model.device)
            h = model(**enc).last_hidden_state[:, 0]
            h = torch.nn.functional.normalize(h.float(), dim=-1)
            out.extend(h.cpu().numpy().astype(np.float16))
    del model
    torch.cuda.empty_cache()
    return out


def encode_queries_st(model_path, questions, batch_size=64):
    """Kênh `e5`: SentenceTransformer, KHÔNG tiền tố (model nền Qwen3, không phải họ e5).

    Guard pooling (result.md §65): checkpoint lưu bằng save_pretrained mà thiếu
    modules.json/1_Pooling thì SentenceTransformer âm thầm dùng MEAN cho câu hỏi trong khi
    corpus được encode last-token — lệch im lặng, đã làm hỏng một bài nộp."""
    import torch
    from sentence_transformers import SentenceTransformer
    if os.path.isdir(model_path) and not os.path.exists(os.path.join(model_path, "modules.json")):
        raise SystemExit(f"⛔ {model_path}: thiếu modules.json/1_Pooling — pooling câu hỏi sẽ "
                         f"lệch với corpus. Chép modules.json, 1_Pooling/, "
                         f"config_sentence_transformers.json, sentence_bert_config.json.")
    model = SentenceTransformer(model_path,
                                device="cuda" if torch.cuda.is_available() else "cpu")
    out = []
    for i in range(0, len(questions), batch_size):
        out.extend(model.encode(questions[i:i + batch_size], normalize_embeddings=True))
    del model
    torch.cuda.empty_cache()
    return out


def encode_queries_e5(model_path, questions, batch_size=64, prefix="query: "):
    """Kênh `e5` của pool đào cặp CE fam (`train_fast`): e5_ft, mean pooling, tiền tố "query: ".

    Tái hiện `retrieval/encoder.py::E5Encoder` (SentenceTransformer, checkpoint e5_ft không có
    modules.json ⇒ ST dựng MEAN pooling — ĐÚNG với corpus `e5ft_dense` encode mean + "passage: ",
    nên ở đây không chặn như `encode_queries_st`)."""
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path,
                                device="cuda" if torch.cuda.is_available() else "cpu")
    out = []
    for i in range(0, len(questions), batch_size):
        out.extend(model.encode([prefix + q for q in questions[i:i + batch_size]],
                                normalize_embeddings=True))
    del model
    torch.cuda.empty_cache()
    return out
