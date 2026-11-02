from pathlib import Path
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


class DocumentSearchEngine:

    def __init__(self):

        self.documents = []

        for path in sorted(DOCUMENTS_DIR.glob("*.txt")):

            text = path.read_text(
                encoding="utf-8-sig"
            ).strip()

            if text:
                self.documents.append({
                    "source": path.name,
                    "text": text,
                })

        if not self.documents:
            raise ValueError("No documents found.")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )

        texts = [
            document["text"]
            for document in self.documents
        ]

        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

        self.tokenized_documents = [
            self.tokenize(text)
            for text in texts
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

    @staticmethod
    def tokenize(text):

        return re.findall(
            r"[a-z0-9]+(?:-[a-z0-9]+)*",
            text.lower(),
        )

    def search(self, query, top_k=3):

        if not query.strip():
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        _, semantic_ids = self.index.search(
            query_embedding,
            len(self.documents),
        )

        keyword_scores = self.bm25.get_scores(
            self.tokenize(query)
        )

        keyword_ids = np.argsort(
            -keyword_scores
        )

        scores = {}

        for rank, document_id in enumerate(
            semantic_ids[0]
        ):
            document_id = int(document_id)

            scores[document_id] = (
                scores.get(document_id, 0.0)
                + 1.0 / (60 + rank + 1)
            )

        for rank, document_id in enumerate(
            keyword_ids
        ):
            document_id = int(document_id)

            if keyword_scores[document_id] <= 0:
                continue

            scores[document_id] = (
                scores.get(document_id, 0.0)
                + 1.0 / (60 + rank + 1)
            )

        ranked_ids = sorted(
            scores,
            key=lambda document_id: -scores[document_id],
        )

        return [
            {
                "source": self.documents[document_id]["source"],
                "text": self.documents[document_id]["text"],
                "score": round(scores[document_id], 5),
            }
            for document_id in ranked_ids[:top_k]
        ]


_engine = None


def get_engine():

    global _engine

    if _engine is None:
        _engine = DocumentSearchEngine()

    return _engine


@tool
def search_documents(query: str) -> str:
    """Search company documents for relevant information."""

    results = get_engine().search(
        query,
        top_k=3,
    )

    if not results:
        return "No relevant documents found."

    return "\n\n".join(
        f"Source: {item['source']}\n"
        f"Content: {item['text']}"
        for item in results
    )


if __name__ == "__main__":

    question = "What does SEC-204 mean?"

    results = get_engine().search(question)

    for item in results:
        print("\nSOURCE:", item["source"])
        print("SCORE:", item["score"])
        print("TEXT:", item["text"][:250])