import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import re
import pickle
import argparse
import glob
import gc
import torch
from tqdm import tqdm
from qdrant_client import models

from ..encoder import E5Encoder, BGEM3Encoder, SentenceEncoder as GTEEncoder
from ..search import Qdrant, BM25

def load_corpus_from_dir(corpus_dir):
    corpus_data = []
    file_paths = sorted(glob.glob(os.path.join(corpus_dir, "*.json")))
    print(f"Tìm thấy {len(file_paths)} file JSON trong thư mục '{corpus_dir}'. Bắt đầu đọc...")
    for file_path in tqdm(file_paths, desc="Loading JSON files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    corpus_data.append(data)
                elif isinstance(data, list):
                    corpus_data.extend(data)
        except Exception as e:
            pass
    return corpus_data

def count_tokens(text): return len(text.split())

def get_last_overlap(text):
    section_pattern = r'[^.!?]+[.!?]'
    sections = re.findall(section_pattern, text.strip(), flags=re.DOTALL)
    if sections: return sections[-1].strip()
    paras = re.split(r'\n\s*\n', text.strip())
    if len(paras) > 0: return paras[-1].strip()
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    if len(sentences) > 0: return sentences[-1].strip()
    return text.strip()

def split_chunks(text, max_len=2000):
    section_pattern = r'[^.!?]+[.!?]'
    sections = re.findall(section_pattern, text, flags=re.DOTALL)
    if not sections: sections = [text.strip()]
    raw_chunks, current_chunk_content = [], ""
    for section in sections:
        section = section.strip()
        tokens_to_add = count_tokens(section) + (1 if current_chunk_content else 0)
        if count_tokens(current_chunk_content) + tokens_to_add <= max_len:
            current_chunk_content = current_chunk_content + "\n" + section if current_chunk_content else section
        else:
            if current_chunk_content: raw_chunks.append(current_chunk_content)
            if count_tokens(section) > max_len:
                paras = re.split(r'\n\s*\n', section)
                sub_chunk_content = ""
                for para in paras:
                    para = para.strip()
                    para_tokens_to_add = count_tokens(para) + (2 if sub_chunk_content else 0)
                    if count_tokens(sub_chunk_content) + para_tokens_to_add <= max_len:
                        sub_chunk_content = sub_chunk_content + "\n\n" + para if sub_chunk_content else para
                    else:
                        if sub_chunk_content: raw_chunks.append(sub_chunk_content)
                        sub_chunk_content = para
                if sub_chunk_content: raw_chunks.append(sub_chunk_content)
                current_chunk_content = ""
            else:
                current_chunk_content = section
    if current_chunk_content: raw_chunks.append(current_chunk_content)
    final_chunks = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0: final_chunks.append(chunk)
        else:
            overlap = get_last_overlap(raw_chunks[i - 1])
            current_overlap = overlap
            while True:
                potential_new_chunk = (current_overlap + "\n\n" + chunk).strip()
                if count_tokens(potential_new_chunk) <= max_len:
                    final_chunks.append(potential_new_chunk)
                    break
                if len(current_overlap.split()) > 1: current_overlap = ' '.join(current_overlap.split()[1:])
                elif len(current_overlap) > 0: current_overlap = current_overlap[1:]
                else:
                    final_chunks.append(chunk)
                    break
    return final_chunks

def batch_encode(encoder, texts, batch_size, encode_type="passage"):
    dense_all, indices_all, values_all = [], [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding in batch"):
        batch_texts = texts[i:i + batch_size]
        dense, indices, values = encoder.encode(batch_texts, type=encode_type)
        dense_all.extend(dense)
        indices_all.extend(indices)
        values_all.extend(values)
    return dense_all, indices_all, values_all

def clean_texts(texts, payloads, ids):
    clean_t, clean_p, clean_i = [], [], []
    for i, text in enumerate(texts):
        if isinstance(text, str) and text.strip():
            clean_t.append(text.strip())
            clean_p.append(payloads[i])
            clean_i.append(ids[i])
    return clean_t, clean_p, clean_i

def main(args):
    corpus_data = load_corpus_from_dir(args.corpus_path)
    if not corpus_data: return

    corpus_for_qdrant = []
    point_id_counter = 0

    for index, row in enumerate(corpus_data):
        original_aid = row.get('id', f"unknown_{index}")
        context_text = str(row.get('passage', ''))
        if not context_text.strip(): continue
        chunks = split_chunks(context_text, max_len=args.max_chunk_length)
        for chunk_idx, chunk_text in enumerate(chunks):
            point_id_counter += 1
            chunk_identifier = f"{original_aid}_{index+1}_chunk{chunk_idx+1}"
            corpus_for_qdrant.append({
                "id": point_id_counter,
                "text": chunk_text,
                "metadata": {
                    "context": chunk_text,
                    "aid": original_aid,
                    "chunk_identifier": chunk_identifier,
                    "link": row.get('link', ''),
                    "name": row.get('name', '')
                }
            })

    texts_to_encode = [item['text'] for item in corpus_for_qdrant]
    payloads = [item['metadata'] for item in corpus_for_qdrant]
    ids = [item['id'] for item in corpus_for_qdrant]
    texts_to_encode, payloads, ids = clean_texts(texts_to_encode, payloads, ids)

    if not args.skip_bm25:
        print("\n--- [0/3] ĐANG XÂY DỰNG CHỈ MỤC BM25 ---")
        bm25_index_corpus = [p["chunk_identifier"] for p in payloads]
        bm25 = BM25(corpus=texts_to_encode, index_corpus=bm25_index_corpus)
        with open(args.bm25_index_path, "wb") as f: pickle.dump(bm25, f)
        chunk_to_aid = {p["chunk_identifier"]: p["aid"] for p in payloads}
        with open(args.bm25_meta_path, "w", encoding="utf-8") as f: json.dump(chunk_to_aid, f, ensure_ascii=False)
        print("✅ Đã lưu BM25 thành công!")
    else:
        print(f"\n⏩ Bỏ qua BM25 theo yêu cầu.")

    UPLOAD_BATCH_SIZE = 10000

    if args.target_model in ["bge", "all"]:
        print(f"\n--- [1/3] BẮT ĐẦU ENCODE MÔ HÌNH BGE-M3 ({len(texts_to_encode)} chunks) ---")
        bge_encoder = BGEM3Encoder(model_name=args.bge_model_name)
        bge_qdrant = Qdrant(path=args.bge_qdrant_path, collection_name=args.bge_collection_name)

        current_points = 0
        try:
            info = bge_qdrant.client.get_collection(bge_qdrant.collection_name)
            current_points = info.points_count
            print(f"🔄 Tìm thấy dữ liệu cũ: Đã có {current_points} chunks trong Qdrant.")
        except Exception:
            print("✨ Tạo collection mới...")
            bge_qdrant.create_collection(dimension=bge_encoder.get_dimension())

        for start_idx in range(current_points, len(texts_to_encode), UPLOAD_BATCH_SIZE):
            end_idx = min(start_idx + UPLOAD_BATCH_SIZE, len(texts_to_encode))
            print(f"\n  -> BGE-M3: Lô chunk {start_idx} đến {end_idx}")

            batch_texts = texts_to_encode[start_idx:end_idx]
            batch_payloads = payloads[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]

            dense_vecs, indices_list, values_list = batch_encode(bge_encoder, batch_texts, args.batch_size)
            points = []
            for i in range(len(batch_texts)):
                sparse_data = {'indices': indices_list[i], 'values': values_list[i]}
                points.append(models.PointStruct(
                    id=batch_ids[i], vector={"dense": dense_vecs[i], "sparse": sparse_data}, payload=batch_payloads[i]
                ))

            bge_qdrant.add_points(points=points)

            del dense_vecs, indices_list, values_list, points, batch_texts, batch_payloads, batch_ids
            gc.collect()

        print("✅ Đã xong BGE-M3. Xả VRAM...")
        del bge_encoder, bge_qdrant
        gc.collect()
        torch.cuda.empty_cache()

    if args.target_model in ["e5", "all"]:
        print(f"\n--- [2/3] BẮT ĐẦU ENCODE MÔ HÌNH E5 ({len(texts_to_encode)} chunks) ---")
        e5_encoder = E5Encoder(model_name=args.e5_model_name)
        e5_qdrant = Qdrant(path=args.e5_qdrant_path, collection_name=args.e5_collection_name)

        current_points = 0
        try:
            info = e5_qdrant.client.get_collection(e5_qdrant.collection_name)
            current_points = info.points_count
            print(f"🔄 Tìm thấy dữ liệu cũ: Đã có {current_points} chunks trong Qdrant E5.")
        except Exception:
            print("✨ Tạo collection E5 mới...")
            e5_qdrant.create_collection(dimension=e5_encoder.get_dimension())

        for start_idx in range(current_points, len(texts_to_encode), UPLOAD_BATCH_SIZE):
            end_idx = min(start_idx + UPLOAD_BATCH_SIZE, len(texts_to_encode))
            print(f"  -> E5: Lô chunk {start_idx} đến {end_idx}")
            batch_texts = texts_to_encode[start_idx:end_idx]
            batch_payloads = payloads[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]

            dense_vecs, _, _ = batch_encode(e5_encoder, batch_texts, args.batch_size)
            points = [models.PointStruct(id=batch_ids[i], vector={"dense": dense_vecs[i]}, payload=batch_payloads[i]) for i in range(len(batch_texts))]
            e5_qdrant.add_points(points=points)

            del dense_vecs, points, batch_texts, batch_payloads, batch_ids
            gc.collect()

        print("✅ Đã xong E5. Xả VRAM...")
        del e5_encoder, e5_qdrant
        gc.collect()
        torch.cuda.empty_cache()

    if args.target_model in ["gte", "all"]:
        print(f"\n--- [3/3] BẮT ĐẦU ENCODE MÔ HÌNH GTE ({len(texts_to_encode)} chunks) ---")
        gte_encoder = GTEEncoder(model_name=args.gte_model_name)
        gte_qdrant = Qdrant(path=args.gte_qdrant_path, collection_name=args.gte_collection_name)

        current_points = 0
        try:
            info = gte_qdrant.client.get_collection(gte_qdrant.collection_name)
            current_points = info.points_count
            print(f"🔄 Tìm thấy dữ liệu cũ: Đã có {current_points} chunks trong Qdrant GTE.")
        except Exception:
            print("✨ Tạo collection GTE mới...")
            gte_qdrant.create_collection(dimension=gte_encoder.get_dimension())

        for start_idx in range(current_points, len(texts_to_encode), UPLOAD_BATCH_SIZE):
            end_idx = min(start_idx + UPLOAD_BATCH_SIZE, len(texts_to_encode))
            print(f"  -> GTE: Lô chunk {start_idx} đến {end_idx}")
            batch_texts = texts_to_encode[start_idx:end_idx]
            batch_payloads = payloads[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]

            dense_vecs, _, _ = batch_encode(gte_encoder, batch_texts, args.batch_size)
            points = [models.PointStruct(id=batch_ids[i], vector={"dense": dense_vecs[i]}, payload=batch_payloads[i]) for i in range(len(batch_texts))]
            gte_qdrant.add_points(points=points)

            del dense_vecs, points, batch_texts, batch_payloads, batch_ids
            gc.collect()

        print("✅ Đã xong GTE. Xả VRAM...")
        del gte_encoder, gte_qdrant
        gc.collect()
        torch.cuda.empty_cache()

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--bge_model_name", type=str, default="BAAI/bge-m3")
    parser.add_argument("--e5_model_name", type=str, default="intfloat/multilingual-e5-large")
    parser.add_argument("--gte_model_name", type=str, default="Alibaba-NLP/gte-multilingual-base")
    parser.add_argument("--bge_qdrant_path", type=str, default="./bge_qdrant")
    parser.add_argument("--e5_qdrant_path", type=str, default="./e5_qdrant")
    parser.add_argument("--gte_qdrant_path", type=str, default="./gte_qdrant")
    parser.add_argument("--bge_collection_name", type=str, default="bge_corpus")
    parser.add_argument("--e5_collection_name", type=str, default="e5_corpus")
    parser.add_argument("--gte_collection_name", type=str, default="gte_corpus")
    parser.add_argument("--bm25_index_path", type=str, default="./bm25_index.pkl")
    parser.add_argument("--bm25_meta_path", type=str, default="./bm25_meta.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_chunk_length", type=int, default=450)

    parser.add_argument("--target_model", type=str, choices=["bge", "e5", "gte", "all"], default="all")
    parser.add_argument("--skip_bm25", action="store_true")

    return parser

if __name__ == "__main__":
    main(arg_parser().parse_args())
