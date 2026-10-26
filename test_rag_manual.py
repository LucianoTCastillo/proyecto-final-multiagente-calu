import asyncio

from app.rag.hybrid import create_hybrid_retriever
from app.rag.schemas import SearchRequest


async def main():
    retriever = await create_hybrid_retriever()

    queries = [
        "¿Qué datos necesito para un presupuesto?",
        "¿Qué cuidados necesita una pieza de PLA?",
        "¿Pueden confirmar una fecha de entrega?",
        "¿Tienen colores disponibles?",
        "¿Cómo se programa un satélite?",
    ]

    for query in queries:
        result = await retriever.search(
            SearchRequest(query=query, top_k=2)
        )

        print(f"\nConsulta: {query}")

        if not result.hits:
            print("Sin coincidencias documentales.")
            continue

        for hit in result.hits:
            print(
                f"Fuente: {hit.source} | "
                f"RRF: {hit.score:.4f}"
            )
            print(f"Fragmento: {hit.content[:180]}...")


if __name__ == "__main__":
    asyncio.run(main())