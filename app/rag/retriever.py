import asyncio
import math
import re
import unicodedata
from collections import Counter

from app.rag.documents import load_documents
from app.rag.schemas import (
    DocumentChunk,
    SearchHit,
    SearchRequest,
    SearchResponse,
)


STOPWORDS = {
    "a", "al", "algo", "como", "con", "cual", "cuales",
    "de", "del", "el", "ella", "ellos", "en", "es",
    "esta", "estas", "este", "estos", "la", "las",
    "lo", "los", "me", "mi", "o", "para", "pero",
    "por", "que", "quien", "se", "sin", "su", "sus",
    "te", "tu", "un", "una", "unos", "unas", "y",
}


def tokenize(text: str) -> list[str]:
    """Normaliza mayúsculas y acentos y extrae palabras."""
    normalized = unicodedata.normalize("NFD", text.lower())

    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )

    return [
        word
        for word in re.findall(r"[a-z0-9]+", normalized)
        if word not in STOPWORDS
    ]


class BM25Retriever:
    """Índice léxico en memoria para los documentos cargados."""

    def __init__(self, chunks: list[DocumentChunk]):
        if not chunks:
            raise ValueError("Se necesita al menos un fragmento.")

        self.chunks = chunks

        self.tokens = [
            tokenize(f"{chunk.title}\n{chunk.content}")
            for chunk in chunks
        ]

        self.frequencies = [
            Counter(tokens) for tokens in self.tokens
        ]

        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = (
            sum(self.lengths) / len(self.lengths)
        ) or 1.0

        self.document_frequencies = Counter()

        for tokens in self.tokens:
            self.document_frequencies.update(set(tokens))

    def _search(self, request: SearchRequest) -> SearchResponse:
        """Calcula BM25 y devuelve los mejores resultados."""
        query_terms = set(tokenize(request.query))
        hits = []

        k1 = 1.5
        b = 0.75
        document_count = len(self.chunks)

        for chunk, frequencies, length in zip(
            self.chunks,
            self.frequencies,
            self.lengths,
        ):
            score = 0.0

            for term in query_terms:
                frequency = frequencies.get(term, 0)

                if frequency == 0:
                    continue

                matching_documents = (
                    self.document_frequencies[term]
                )

                inverse_frequency = math.log(
                    1
                    + (
                        document_count - matching_documents + 0.5
                    )
                    / (matching_documents + 0.5)
                )

                denominator = frequency + k1 * (
                    1 - b + b * length / self.average_length
                )

                score += (
                    inverse_frequency
                    * frequency
                    * (k1 + 1)
                    / denominator
                )

            if score > 0:
                hits.append(
                    SearchHit(
                        **chunk.model_dump(),
                        score=score,
                    )
                )

        hits.sort(key=lambda hit: (-hit.score, hit.chunk_id))

        return SearchResponse(
            query=request.query,
            hits=hits[:request.top_k],
        )

    async def search(
        self,
        request: SearchRequest,
    ) -> SearchResponse:
        """Ejecuta la búsqueda sin bloquear la API."""
        return await asyncio.to_thread(self._search, request)


async def create_retriever() -> BM25Retriever:
    """Carga los documentos y construye el índice."""
    chunks = await load_documents()

    return await asyncio.to_thread(BM25Retriever, chunks)