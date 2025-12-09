# ===================================================
# reranker.py
# Cross-Encoder Reranker (local, rapide, précis)
# ===================================================

from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Document], top_k: int = 50) -> List[Document]:
        if not docs:
            return []

        pairs = [[query, d.page_content] for d in docs]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True
        )

        return [d for _, d in ranked[:top_k]]
