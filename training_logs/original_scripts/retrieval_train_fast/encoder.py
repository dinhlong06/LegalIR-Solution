import os

from FlagEmbedding import BGEM3FlagModel, FlagReranker
from sentence_transformers import SentenceTransformer
import torch

from typing import List, Tuple, Union


def _pick_device() -> str:
    """cuda nếu có, không thì cpu.

    Trước đây hardcode "cuda:0" nên máy không thấy CUDA là raise ngay lúc dựng object.
    rule.md dòng 60 cho phép BTC đòi log + cấu hình môi trường trong 48h, máy kiểm tra
    không chắc có GPU. Dùng đúng cách chọn của DenseMatrix trong fast_index.py.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


class BGEM3Encoder:
    def __init__(self, model_name: str, use_fp16: bool = True):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.use_fp16 = use_fp16

    def encode(self, query_text: Union[str, List[str]], type: str = "query"):
        is_batch = isinstance(query_text, list)
        emb = self.model.encode(query_text, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        emb_sparse = emb['lexical_weights']
        dense_vec = emb['dense_vecs']

        if not is_batch:
            indices = list(emb_sparse.keys())
            values = list(emb_sparse.values())

            return dense_vec, indices, values
        else:
            dense_out, indices_out, values_out = [], [], []
            for dense_vec, sparse_vec in zip(dense_vec, emb_sparse):
                dense_out.append(dense_vec)
                indices_out.append(list(sparse_vec.keys()))
                values_out.append(list(sparse_vec.values()))
            return dense_out, indices_out, values_out

    def get_dimension(self) -> int:
        dense_vec, _, _ = self.encode("dummy input")
        return dense_vec.shape[0]

class E5Encoder:
    def __init__(self, model_name: str, use_prefix: bool = True,
                 segment: bool = False):
        """use_prefix: ho e5 doi tien to "query: "/"passage: ". Model nen PhoBERT
        (bkai-foundation-models/vietnamese-bi-encoder) KHONG dung tien to nao — nhet vao
        la them nhieu vo nghia. segment: tach tu tieng Viet truoc khi encode, cung cho
        nhom PhoBERT vi chung duoc pretrain tren van ban da tach tu.

        GUARD POOLING (bai hoc result.md §65, 13/09): checkpoint local luu bang
        save_pretrained KHONG kem modules.json/1_Pooling thi SentenceTransformer TU DUNG
        mean pooling cho cau hoi — lech IM LANG voi corpus encode last-token
        (encode_corpus_harrier.py), da lam hong bai all7k_full public. Thieu file cau
        hinh pooling => dung on ao. Escape cho thi nghiem cu co chu dich (pooling mean
        that su): E5_ALLOW_NO_POOLING_CFG=1."""
        if os.path.isdir(model_name) and not os.path.exists(
                os.path.join(model_name, "modules.json")):
            if os.environ.get("E5_ALLOW_NO_POOLING_CFG") != "1":
                raise SystemExit(
                    f"⛔ {model_name}: thiếu modules.json/1_Pooling — SentenceTransformer "
                    f"sẽ âm thầm dùng MEAN pooling cho câu hỏi, trong khi corpus có thể "
                    f"được encode last-token (lỗi §65 làm hỏng bài all7k_full). "
                    f"Vá: chép modules.json, 1_Pooling/, config_sentence_transformers.json, "
                    f"sentence_bert_config.json từ eval/ckpt/harrier_ft/epoch2 (last-token), "
                    f"hoặc từ bản mean-pooling nếu corpus của model này encode mean. "
                    f"Escape có chủ đích: E5_ALLOW_NO_POOLING_CFG=1.")
        self.model = SentenceTransformer(model_name, device=_pick_device())
        self.use_prefix = use_prefix
        self._seg = None
        if segment:
            from underthesea import word_tokenize
            self._seg = lambda s: word_tokenize(s, format="text")
        # In pooling ra log de lech pooling KHONG BAO GIO im lang nua (dau hieu §65:
        # top-30 trung kenh khac chi 17% ma khong ai thay).
        for m in self.model:
            if m.__class__.__name__ == "Pooling":
                mode = ("cls" if getattr(m, "pooling_mode_cls_token", False) else
                        "last_token" if getattr(m, "pooling_mode_lasttoken", False) else
                        "mean" if getattr(m, "pooling_mode_mean_tokens", False) else
                        "max" if getattr(m, "pooling_mode_max_tokens", False) else "?")
                print(f"E5Encoder: {model_name} pooling_cau_hoi={mode}")

    def encode(self, query_text: Union[str, List[str]], type: str = "query"):
        if isinstance(query_text, str):
            query_text = [query_text]
            single = True
        else:
            single = False
        if self._seg is not None:
            query_text = [self._seg(t) for t in query_text]
        if self.use_prefix:
            query_text = [f"{type}: " + t for t in query_text]
        if single:
            query_text = query_text[0]
        dense_vec = self.model.encode(query_text, normalize_embeddings=True)
        return dense_vec, [], []

    def get_dimension(self) -> int:
        dense_vec, _, _ = self.encode("dummy input")
        return dense_vec.shape[0]

class E5InstructEncoder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device=_pick_device())
        self.task = "Với một truy vấn về luật Việt Nam, truy xuất các đoạn văn liên quan có chứa câu trả lời cho truy vấn đó."

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    def encode(self, query_text: Union[str, List[str]], type: str = "query"):
        if not isinstance(query_text, str):
            if type == "query":
                query_text = [self.get_detailed_instruct(self.task, text) for text in query_text]
        else:
            if type == "query":
                query_text = self.get_detailed_instruct(self.task, query_text)
        dense_vec = self.model.encode(query_text, normalize_embeddings=True)
        return dense_vec, [], []

    def get_dimension(self) -> int:
        dense_vec, _, _ = self.encode("dummy input")
        return dense_vec.shape[0]

class SentenceEncoder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True
        )
        self.model.max_seq_length = 512

    def encode(self, query_text: Union[str, List[str]], type: str = "query"):
        dense_vec = self.model.encode(query_text, normalize_embeddings=True)
        return dense_vec, [], []

    def get_dimension(self) -> int:
        dense_vec, _, _ = self.encode("dummy input")
        return dense_vec.shape[0]

class VietEmbedEncoder:
    def __init__(self, model_name: str = "VietAI/vietembed-large"):
        self.model = SentenceTransformer(model_name, device=_pick_device())

    def encode(self, query_text: Union[str, List[str]], type: str = "query"):
        dense_vec = self.model.encode(query_text, normalize_embeddings=True)
        return dense_vec, [], []

    def get_dimension(self) -> int:
        dense_vec, _, _ = self.encode("dummy input")
        return dense_vec.shape[0]


class BGEM3Reranker:
    def __init__(self, model_name: str, use_fp16: bool = True):
        self.model = FlagReranker(model_name, use_fp16=use_fp16)

    def calculate_scores(self, pairs:List[List[str]]) -> List[float]:
        if not isinstance(pairs, list):
            pairs = [pairs]
        scores = self.model.compute_score(pairs, normalize=True)
        return scores

    def rerank(self, query: str, passages: List[Tuple[str, str]], top_k: int = -1) -> List[Tuple[float, str, str]]:
        pairs = [[query, passage[1]] for passage in passages]
        scores = self.calculate_scores(pairs)

        scored_passages = [(score, passage[0], passage[1]) for passage, score in zip(passages, scores)]

        ranked_passages = sorted(scored_passages, key=lambda x: x[0], reverse=True)

        if top_k > 0:
            ranked_passages = ranked_passages[:top_k]

        return ranked_passages

class BGELLMReranker:
    def __init__(self, model_name: str='BAAI/bge-reranker-v2-minicpm-layerwise', use_fp16: bool = True):
        # import lười: chỉ class (đã đóng) này dùng LayerWise, đừng bắt mọi caller
        # của encoder.py trả chi phí import.
        from FlagEmbedding import LayerWiseFlagLLMReranker
        self.model = LayerWiseFlagLLMReranker(model_name, use_fp16=use_fp16)

    def calculate_scores(self, pairs:List[List[str]]) -> List[float]:
        if not isinstance(pairs, list):
            pairs = [pairs]
        scores = self.model.compute_score(pairs, cutoff_layers=[28])
        return scores

    def rerank(self, query: str, passages: List[Tuple[str, str]], top_k: int = -1) -> List[Tuple[float, str, str]]:
        pairs = [[query, passage[1]] for passage in passages]
        scores = self.calculate_scores(pairs)

        scored_passages = [(score, passage[0], passage[1]) for passage, score in zip(passages, scores)]

        ranked_passages = sorted(scored_passages, key=lambda x: x[0], reverse=True)

        if top_k > 0:
            ranked_passages = ranked_passages[:top_k]

        return ranked_passages
