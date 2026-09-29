import os
import time
import numpy as np
import torch
from typing import Union, List, Tuple
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import PointsSelector, PointIdsList
from rank_bm25 import BM25Okapi
import sentencepiece as spm


from .encoder import E5Encoder, BGEM3Encoder

class Qdrant:
    def __init__(self, path: str, collection_name: str):
        self.client = QdrantClient(path=path)
        self.collection_name = collection_name

    def create_collection(self, dimension: int):
        dense_vector_config = {
            "dense": models.VectorParams(size=dimension, distance=models.Distance.COSINE)
        }
        sparse_vector_config = {
            "sparse": models.SparseVectorParams()
        }

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=dense_vector_config,
                sparse_vectors_config=sparse_vector_config
            )
    def delete_collection(self):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

    def add_points(self, points: List[models.PointStruct]):
        if not self.client.collection_exists(self.collection_name):
            raise ValueError(f"Collection {self.collection_name} does not exist.")
        self.client.upload_points(
            collection_name=self.collection_name,
            points=points
        )

    def delete_points(self, ids: List[int]):
        if not self.client.collection_exists(self.collection_name):
            raise ValueError(f"Collection {self.collection_name} does not exist.")
        self.client.delete_points(
            collection_name=self.collection_name,
            ids=ids
        )

    def get_collection_info(self):
        if not self.client.collection_exists(self.collection_name):
            raise ValueError(f"Collection {self.collection_name} does not exist.")
        return self.client.get_collection(self.collection_name)

    def get_points(self, ids: List[int]) -> List[models.PointStruct]:
        if not self.client.collection_exists(self.collection_name):
            raise ValueError(f"Collection {self.collection_name} does not exist.")
        return self.client.get_points(
            collection_name=self.collection_name,
            ids=ids
        )
    def search(self, query: str, encode: Union[E5Encoder, BGEM3Encoder], top_k: int = 10, hybrid: bool = True):
        dense_vec, indices, values = encode.encode(query, type="query")

        prefetch = [
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=top_k,
            )
        ]
        if hybrid and indices and values:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=indices,
                        values=values
                    ),
                    using="sparse",
                    limit=top_k,
                ),
            )

        results = self.client.query_points(
            self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            with_payload=True,
            limit= top_k,
        )
        return results

    def batch_search(self, queries: List[str], encode: Union[E5Encoder, BGEM3Encoder], top_k: int = 10):
        dense_vecs, indices_list, values_list = encode.encode(queries, type="query")

        results_all = []
        scores_all = []
        for i in range(len(queries)):
            dense_vec = dense_vecs[i]

            prefetch = [
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=top_k,
                )
            ]

            if indices_list and values_list and indices_list[i] and values_list[i]:
                indices = [int(x) for x in indices_list[i]]
                values = [float(x) for x in values_list[i]]
                prefetch.append(
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=indices,
                            values=values
                        ),
                        using="sparse",
                        limit=top_k,
                    )
                )

            result = self.client.query_points(
                self.collection_name,
                prefetch=prefetch,
                query=models.FusionQuery(
                    fusion=models.Fusion.RRF,
                ),
                with_payload=True,
                limit=top_k,
            ).points
            single_result = {
                "context": [point.payload["context"] for point in result],
                "aid": [point.payload["aid"] for point in result]
            }
            scores_all.append([point.score for point in result])
            results_all.append(single_result)

        return results_all, scores_all

    def get_ids(self, ids: Union[List[int], List[str]]):
        if not self.client.collection_exists(self.collection_name):
            raise ValueError(f"Collection {self.collection_name} does not exist.")
        return self.client.get_points(
            collection_name=self.collection_name,
            ids=ids
        )

    def search_by_ids(self, query: str, encode: Union[E5Encoder, BGEM3Encoder], aids: List[str], top_k: int = 50):
        dense_vec, indices, values = encode.encode(query, type="query")

        prefetch = [
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=top_k,
            )
        ]
        if indices and values:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=indices,
                        values=values
                    ),
                    using="sparse",
                    limit=top_k,
                ),
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            points=PointsSelector(
                points=PointIdsList(
                    ids=aids
                )
            ),
            with_payload=True,
            limit=top_k,
        )
        return results


    def search_by_aids(self, query: str, encode, aids: List[str], top_k: int = 50):
        dense_vec, indices, values = encode.encode(query, type="query")
        if not self.client.collection_exists(self.collection_name):
            raise ValueError(f"Collection {self.collection_name} does not exist.")

        if top_k == -1:
            top_k = len(aids)
        prefetch = [
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=top_k,
            )
        ]
        if indices and values:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=indices,
                        values=values
                    ),
                    using="sparse",
                    limit=top_k,
                ),
            )

        results = self.client.query_points(
            self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="aid",
                        match=models.MatchAny(any=aids)
                    )
                ]
            ),
            with_payload=True,
            limit= top_k,
        )
        return results

# Định nghĩa đã chuyển sang fast_index.py để đường inference nhanh không phải import
# file này (nó kéo theo qdrant_client). Giữ tên ở đây cho code cũ và verify_fast_index.py.
from .fast_index import _top_k_indices  # noqa: E402,F401


class BM25:
    def __init__(self, corpus: List[str], index_corpus: List[str]):
        self.tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.index_corpus = index_corpus
        self.corpus = corpus

    def get_scores(self, queries: List[str]) -> List[List[float]]:
        return [self.bm25.get_scores(query.split()) for query in queries]

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, str, str]]:
        scores = self.bm25.get_scores(query.split())
        top_k_indices = _top_k_indices(scores, top_k)
        return [(scores[i], self.index_corpus[i], self.corpus[i]) for i in top_k_indices]


class VectorizedBM25:
    """Vectorized BM25 using scipy sparse matrix multiplication.

    Precomputes a (n_terms × n_docs) sparse matrix of TF-normalized weights.
    Scoring a batch of queries is then just Q @ W — orders of magnitude faster
    than calling rank_bm25.get_scores() per query.
    """

    def __init__(self, bm25_okapi, corpus: List[str], index_corpus: List[str]):
        from scipy import sparse as sp_sparse
        self.corpus = corpus
        self.index_corpus = index_corpus
        self.k1 = bm25_okapi.k1
        self.b = bm25_okapi.b
        self.avgdl = bm25_okapi.avgdl
        self.doc_len = bm25_okapi.doc_len

        # Build term→index mapping
        idf = bm25_okapi.idf
        self.term2idx = {term: idx for idx, term in enumerate(idf.keys())}
        self.idf_arr = np.array([idf[t] for t in self.term2idx.keys()], dtype=np.float32)
        self.n_terms = len(self.term2idx)
        self.n_docs = len(bm25_okapi.doc_freqs)

        # Build sparse term-document matrix W[t,d] = tf_norm(t,d)
        print(f"   Building sparse BM25 matrix ({self.n_terms} terms × {self.n_docs} docs)...")
        t0 = time.time()
        rows, cols, data = [], [], []
        for doc_idx, doc_freqs in enumerate(bm25_okapi.doc_freqs):
            dl_norm = 1.0 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl
            for term, tf in doc_freqs.items():
                if term in self.term2idx:
                    rows.append(self.term2idx[term])
                    cols.append(doc_idx)
                    data.append(tf * (self.k1 + 1.0) / (tf + self.k1 * dl_norm))

        self.W = sp_sparse.csr_matrix(
            (np.float32(data), (np.int32(rows), np.int32(cols))),
            shape=(self.n_terms, self.n_docs)
        )
        print(f"   Sparse matrix built in {time.time()-t0:.1f}s ({self.W.nnz} nonzeros)")

    def _build_query_matrix(self, tokenized_queries: List[List[str]]):
        """Build (n_queries × n_terms) sparse matrix Q where Q[i,t] = IDF(t) if t in query_i."""
        from scipy import sparse as sp_sparse
        q_rows, q_cols, q_data = [], [], []
        for q_idx, tokens in enumerate(tokenized_queries):
            for term in tokens:
                tidx = self.term2idx.get(term)
                if tidx is not None:
                    q_rows.append(q_idx)
                    q_cols.append(tidx)
                    q_data.append(self.idf_arr[tidx])
        Q = sp_sparse.csr_matrix(
            (np.float32(q_data), (np.int32(q_rows), np.int32(q_cols))),
            shape=(len(tokenized_queries), self.n_terms)
        )
        return Q

    def batch_search(self, queries: List[str], top_k: int = 200, batch_size: int = 256):
        """Score all queries at once and return top_k results per query.

        Returns:
            results: List of List of (score, index_corpus, corpus) per query
        """
        tokenized = [q.split() for q in queries]
        all_results = []

        for start in range(0, len(queries), batch_size):
            end = min(start + batch_size, len(queries))
            Q = self._build_query_matrix(tokenized[start:end])
            # scores shape: (batch_size, n_docs)
            scores = Q.dot(self.W).toarray()

            for i in range(end - start):
                row_scores = scores[i]
                top_k_idx = _top_k_indices(row_scores, top_k)
                results = [(float(row_scores[j]), self.index_corpus[j], self.corpus[j]) for j in top_k_idx]
                all_results.append(results)

        return all_results

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, str, str]]:
        results = self.batch_search([query], top_k=top_k)
        return results[0] if results else []

class BM25WithBPE:
    def __init__(self, corpus: List[str], index_corpus: List[str], sp_model_path: str):
        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.tokenized_corpus = [self.sp.encode(doc, out_type=str) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.index_corpus = index_corpus
        self.corpus = corpus

    def get_scores(self, queries: List[str]) -> List[List[float]]:
        return [self.bm25.get_scores(self.sp.encode(query, out_type=str)) for query in queries]

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, int, str]]:
        query_tokens = self.sp.encode(query, out_type=str)
        scores = self.bm25.get_scores(query_tokens)
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(scores[i], self.index_corpus[i], self.corpus[i]) for i in top_k_indices]
