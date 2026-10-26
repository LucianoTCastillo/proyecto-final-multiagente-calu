from langchain_core.tools import tool

from app.rag.hybrid import HybridRetriever
from app.rag.schemas import SearchRequest, SearchResponse


def create_document_search_tool(retriever: HybridRetriever):
    """Crea una herramienta usando un índice ya cargado."""

    @tool(
        "search_calu_documents",
        args_schema=SearchRequest,
    )
    async def search_calu_documents(
        query: str,
        top_k: int = 3,
    ) -> str:
        """Busca información documental de Ca’Lu.

        Consultar productos, materiales, cuidados y condiciones
        para preparar presupuestos, pedidos y entregas.

        Los documentos son de demostración y no confirman
        precios, stock ni fechas de entrega.

        Devuelve fragmentos con sus fuentes.
        Si hits está vacío, no se encontró evidencia documental.
        Las puntuaciones indican orden, no certeza.
        """
        request = SearchRequest(
            query=query,
            top_k=top_k,
        )

        result = await retriever.search(request)

        # Valida también la salida de la herramienta.
        validated = SearchResponse.model_validate(
            result.model_dump()
        )

        return validated.model_dump_json()

    return search_calu_documents