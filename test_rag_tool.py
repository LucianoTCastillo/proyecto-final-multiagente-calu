import asyncio

from pydantic import ValidationError

from app.rag.hybrid import create_hybrid_retriever
from app.rag.schemas import SearchResponse
from app.rag.tools import create_document_search_tool


async def main():
    retriever = await create_hybrid_retriever()
    search_tool = create_document_search_tool(retriever)

    queries = [
        "¿Qué datos necesito para un presupuesto?",
        "¿Qué cuidados necesita una pieza de PLA?",
        "¿Cómo se programa un satélite?",
    ]

    for query in queries:
        raw_result = await search_tool.ainvoke(
            {
                "query": query,
                "top_k": 5,
            }
        )

        result = SearchResponse.model_validate_json(raw_result)

        print(f"\nCONSULTA: {result.query}")

        if not result.hits:
            print("Sin coincidencias documentales.")

        for position, hit in enumerate(result.hits, start=1):
            heading = hit.content.splitlines()[0]

            print(
                f"{position}. {hit.source} | "
                f"{heading} | Puntaje: {hit.score:.4f}"
            )

    # Comprueba que la herramienta rechace consultas vacías.
    try:
        await search_tool.ainvoke(
            {"query": "   ", "top_k": 3}
        )
    except ValidationError:
        print("\nOK: la herramienta rechazó la consulta vacía.")
    else:
        raise AssertionError(
            "La herramienta aceptó una consulta vacía."
        )


if __name__ == "__main__":
    asyncio.run(main())