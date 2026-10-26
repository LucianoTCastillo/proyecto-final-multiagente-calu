import asyncio
import json
from decimal import Decimal
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import ValidationError

from app.config import get_llm
from app.sales_schemas import (
    CalculateSalesRequest,
    SalesSearchResponse,
    sales_analysis_adapter,
)
from app.state import OrchestratorState


def validated_analysis(
    results: dict[str, Any],
) -> dict[str, Any]:
    """Valida el formato de salida del cálculo."""
    return sales_analysis_adapter.validate_python(
        results
    ).model_dump()


def analyze_sales(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Valida los datos y calcula la facturación."""
    try:
        validated = SalesSearchResponse.model_validate(
            {"records": data}
        )
    except ValidationError as exc:
        errors = [
            (
                f"{'.'.join(str(part) for part in error['loc'])}: "
                f"{error['msg']}"
            )
            for error in exc.errors()
        ]

        return validated_analysis({
            "valid": False,
            "errors": errors,
        })

    records = validated.records

    if not records:
        return validated_analysis({
            "valid": False,
            "errors": ["No hay datos de ventas para analizar."],
        })

    errors = []
    products = []
    seen_products = set()

    for record in records:
        product = record.product

        if product in seen_products:
            errors.append(f"Producto duplicado: {product}.")
            continue

        seen_products.add(product)

        price = Decimal(str(record.unit_price))
        revenue = price * record.units

        products.append({
            "product": product,
            "units": record.units,
            "unit_price": str(price),
            "revenue": str(revenue),
            "currency": "ARS",
            "source": record.source,
        })

    # No devolver un total parcial si hay registros inválidos.
    if errors:
        return validated_analysis({
            "valid": False,
            "errors": errors,
        })

    total_revenue = sum(
        (Decimal(item["revenue"]) for item in products),
        Decimal("0"),
    )

    highest_revenue = max(
        Decimal(item["revenue"])
        for item in products
    )

    leaders = [
        item["product"]
        for item in products
        if Decimal(item["revenue"]) == highest_revenue
    ]

    return validated_analysis({
        "valid": True,
        "errors": [],
        "products": products,
        "total_units": sum(
            item["units"]
            for item in products
        ),
        "total_revenue": str(total_revenue),
        "top_products": leaders,
        "highest_revenue": str(highest_revenue),
        "currency": "ARS",
    })


async def analyst_agent(
    state: OrchestratorState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Analiza ventas con Groq y una herramienta de cálculo."""
    verified_results: dict[str, Any] = {}
    tool_calls: list[str] = []

    @tool(args_schema=CalculateSalesRequest)
    async def calculate_sales() -> str:
        """Valida y analiza los datos recuperados por el investigador.

        Calcula facturación por producto, total de facturación,
        unidades totales y productos líderes.
        Devuelve valid=False si encuentra datos inválidos.
        """
        results = await asyncio.to_thread(
            analyze_sales,
            state.get("research_data", []),
        )

        results = validated_analysis(results)

        verified_results.clear()
        verified_results.update(results)
        tool_calls.append("calculate_sales")

        return json.dumps(
            results,
            ensure_ascii=False,
        )

    agent = create_react_agent(
        model=get_llm(),
        tools=[calculate_sales],
        prompt=(
            "Sos un analista especializado en ventas. "
            "Usá calculate_sales antes de presentar resultados. "
            "No inventes ni modifiques cantidades, precios o cálculos. "
            "Si valid es False, explicá los errores y pedí revisión. "
            "Si valid es True, explicá la facturación por producto, "
            "el total, las unidades totales y los productos líderes. "
            "Respetá los empates. "
            "La facturación no es ganancia: no hay datos de costos. "
            "Los datos son ficticios y la moneda es ARS. "
            "Respondé en español y considerá las observaciones "
            "del supervisor."
        ),
    )

    instruction = (
        f"Solicitud del usuario: {state['user_query']}\n"
        "Datos recuperados:\n"
        f"{json.dumps(state.get('research_data', []), ensure_ascii=False)}\n"
        "Observaciones del supervisor: "
        f"{state.get('validation_feedback', '')}"
    )

    response = await agent.ainvoke(
        {"messages": [HumanMessage(content=instruction)]},
        config={
            **config,
            "recursion_limit": 12,
        },
    )

    if not tool_calls:
        raise RuntimeError(
            "El analista no utilizó la herramienta de cálculo."
        )

    content = response["messages"][-1].content
    summary = (
        content
        if isinstance(content, str)
        else json.dumps(content, ensure_ascii=False)
    )

    feedback = ""

    if not verified_results.get("valid"):
        feedback = "\n".join(
            verified_results.get(
                "errors",
                ["No se obtuvieron resultados verificados."],
            )
        )

        summary = (
            f"Análisis pendiente de validación:\n{feedback}\n\n"
            f"{summary}"
        )

    return {
        "analysis_results": verified_results,
        "analysis_summary": summary,
        "validation_feedback": feedback,
        "contributions": [
            *state.get("contributions", []),
            {
                "agent": "analyst",
                "tool": "calculate_sales",
                "summary": summary,
            },
        ],
        "messages": [
            AIMessage(content=summary, name="analyst")
        ],
        "steps": state.get("steps", 0) + 1,
    }