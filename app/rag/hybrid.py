import asyncio
import os
from threading import Lock

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from app.rag.documents import PROJECT_ROOT, load_documents
from app.rag.retriever import BM25Retriever, tokenize
from app.rag.schemas import (
    DocumentChunk,
    SearchHit,
    SearchRequest,
    SearchResponse,
)


class HybridRetriever:
    """Combina rankings léxicos y semánticos mediante RRF."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        model_name: str,
        min_similarity: float,
    ):
        if not -1 <= min_similarity <= 1:
            raise ValueError(
                "La similitud mínima debe estar entre -1 y 1."
            )

        self.chunks = chunks
        self.lexical = BM25Retriever(chunks)
        self.min_similarity = min_similarity

        # Evita ejecutar varias inferencias simultáneas
        # sobre la misma instancia del modelo.
        self.model_lock = Lock()

        self.model = SentenceTransformer(
            model_name,
            device="cpu",
        )

        self.embeddings = self.model.encode(
            [chunk.content for chunk in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        self.by_id = {
            chunk.chunk_id: chunk for chunk in chunks
        }

    def _search(self, request: SearchRequest) -> SearchResponse:
        # Recuperamos todos los candidatos disponibles.
        lexical_result = self.lexical._search(
            SearchRequest(
                query=request.query,
                top_k=10,
            )
        )

        with self.model_lock:
            query_embedding = self.model.encode(
                [request.query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]

        # Con vectores normalizados, el producto escalar
        # equivale a la similitud del coseno.
        similarities = self.embeddings @ query_embedding

        semantic_indices = [
            int(index)
            for index in np.argsort(-similarities)
            if float(similarities[index]) >= self.min_similarity
        ][:10]

        semantic_ids = [
            self.chunks[index].chunk_id
            for index in semantic_indices
        ]

        lexical_ids = [
            hit.chunk_id for hit in lexical_result.hits
        ]

        # Reciprocal Rank Fusion combina posiciones,
        # sin sumar puntuaciones de escalas diferentes.
        fused_scores: dict[str, float] = {}

        for ranking in (lexical_ids, semantic_ids):
            for position, chunk_id in enumerate(ranking, start=1):
                fused_scores[chunk_id] = (
                    fused_scores.get(chunk_id, 0.0)
                    + 1.0 / (60 + position)
                )
        # Favorece secciones con encabezados específicos
        # relacionados con la consulta.
        query_terms = set(tokenize(request.query))

        for chunk_id in fused_scores:
            chunk = self.by_id[chunk_id]
            heading = chunk.content.splitlines()[0]
            heading_terms = set(tokenize(heading))

            if heading.startswith("#") and heading_terms:
                heading_coverage = (
                    len(query_terms & heading_terms)
                    / len(heading_terms)
                )

                fused_scores[chunk_id] += (
                    0.02 * heading_coverage
                )

        ranked_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (
                -fused_scores[chunk_id],
                chunk_id,
            ),
        )

        hits = [
            SearchHit(
                **self.by_id[chunk_id].model_dump(),
                score=fused_scores[chunk_id],
            )
            for chunk_id in ranked_ids[:request.top_k]
        ]

        return SearchResponse(
            query=request.query,
            hits=hits,
        )

    async def search(
        self,
        request: SearchRequest,
    ) -> SearchResponse:
        return await asyncio.to_thread(self._search, request)


async def create_hybrid_retriever() -> HybridRetriever:
    load_dotenv(PROJECT_ROOT / ".env")

    model_name = os.getenv("EMBEDDING_MODEL", "").strip()

    if not model_name:
        raise ValueError(
            "Falta configurar EMBEDDING_MODEL en .env."
        )

    min_similarity = float(
        os.getenv("RAG_MIN_SIMILARITY", "0.35")
    )

    chunks = await load_documents()

    return await asyncio.to_thread(
        HybridRetriever,
        chunks,
        model_name,
        min_similarity,
    )