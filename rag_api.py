"""
Local RAG paper Q&A demo.

Start with:
    uvicorn rag_api:app --reload --port 8000
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CORPUS_PATH = Path("web/data/rag_corpus.json")
TOP_K = 3

app = FastAPI(title="Paper Daily Local RAG Demo", version="0.1.0")


class AskRequest(BaseModel):
    question: str


class RagIndex:
    def __init__(self, corpus_path: Path) -> None:
        self.corpus_path = corpus_path
        self.records: list[dict[str, Any]] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.error = ""
        self.load()

    def load(self) -> None:
        self.records = []
        self.vectorizer = None
        self.matrix = None
        self.error = ""

        if not self.corpus_path.exists():
            self.error = f"RAG corpus file not found: {self.corpus_path}"
            return

        try:
            data = json.loads(self.corpus_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.error = f"Failed to read RAG corpus: {exc}"
            return

        if not isinstance(data, list):
            self.error = "RAG corpus must be a JSON array."
            return

        self.records = [record for record in data if isinstance(record, dict)]
        if not self.records:
            self.error = "RAG corpus is empty."
            return

        texts = [str(record.get("text_for_embedding") or "") for record in self.records]
        if not any(text.strip() for text in texts):
            self.error = "RAG corpus has no text_for_embedding content."
            return

        # TF-IDF keeps the demo fully local and deterministic; no external embedding API is required.
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self.matrix = self.vectorizer.fit_transform(texts)

    @property
    def ready(self) -> bool:
        return bool(self.vectorizer is not None and self.matrix is not None and not self.error)

    def search(self, question: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        if not self.ready:
            raise RuntimeError(self.error or "RAG index is not ready.")
        if not question.strip():
            raise ValueError("Question must not be empty.")

        assert self.vectorizer is not None
        query_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(query_vector, self.matrix).ravel()
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for index in top_indices:
            record = self.records[int(index)]
            results.append(format_retrieved_paper(record, float(similarities[int(index)])))
        return results


def float_value(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def authors_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(author) for author in value if str(author).strip())
    return str(value or "")


def paper_summary(record: dict[str, Any]) -> str:
    parts = [
        str(record.get("problem") or "").strip(),
        str(record.get("method") or "").strip(),
        str(record.get("innovation") or "").strip(),
    ]
    summary = " ".join(part for part in parts if part)
    return summary or str(record.get("abstract") or "")


def format_retrieved_paper(record: dict[str, Any], similarity: float) -> dict[str, Any]:
    return {
        "title": str(record.get("title") or ""),
        "authors": authors_text(record.get("authors")),
        "final_score": float_value(record.get("final_score")),
        "similarity": round(similarity, 4),
        "source_url": str(record.get("source_url") or ""),
        "pdf_url": str(record.get("pdf_url") or ""),
        "summary": paper_summary(record),
    }


def build_answer(question: str, papers: list[dict[str, Any]]) -> str:
    if not papers:
        return "No relevant papers were retrieved from the local corpus."

    best = papers[0]
    if best["similarity"] <= 0:
        return "No strong lexical match was found. The returned papers are the closest available entries in the local corpus."

    return (
        "Based on the retrieved papers, the most relevant paper is "
        f"'{best['title']}'. It appears related to the question because its indexed title, abstract, "
        "and summary fields have the highest TF-IDF similarity to the query."
    )


rag_index = RagIndex(CORPUS_PATH)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if rag_index.ready else "error",
        "corpus_path": str(CORPUS_PATH),
        "paper_count": len(rag_index.records),
        "error": rag_index.error,
    }


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    if not rag_index.ready:
        raise HTTPException(status_code=503, detail=rag_index.error or "RAG corpus is not ready.")

    try:
        retrieved_papers = rag_index.search(request.question, top_k=TOP_K)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "question": request.question,
        "answer": build_answer(request.question, retrieved_papers),
        "retrieved_papers": retrieved_papers,
    }
