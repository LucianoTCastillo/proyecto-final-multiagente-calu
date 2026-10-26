from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StringConstraints,
    TypeAdapter,
    field_validator,
)


NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class SalesSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: NonEmptyText = Field(max_length=4000)


class SalesRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: NonEmptyText
    units: StrictInt = Field(ge=0)
    unit_price: StrictInt | StrictFloat = Field(
        ge=0,
        allow_inf_nan=False,
    )
    currency: Literal["ARS"]
    source: NonEmptyText


class SalesSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[SalesRecord]


class CalculateSalesRequest(BaseModel):
    """La herramienta toma los registros del estado del grafo."""

    model_config = ConfigDict(extra="forbid")


class ProductAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: NonEmptyText
    units: StrictInt = Field(ge=0)
    unit_price: str
    revenue: str
    currency: Literal["ARS"]
    source: NonEmptyText

    @field_validator("unit_price", "revenue")
    @classmethod
    def validate_amount(cls, value: str) -> str:
        from decimal import Decimal, InvalidOperation

        try:
            amount = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError("El importe debe ser decimal.") from exc

        if not amount.is_finite() or amount < 0:
            raise ValueError("El importe debe ser finito y no negativo.")

        return value


class SalesAnalysisSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: Literal[True]
    errors: list[str] = Field(max_length=0)
    products: list[ProductAnalysis] = Field(min_length=1)
    total_units: StrictInt = Field(ge=0)
    total_revenue: str
    top_products: list[NonEmptyText] = Field(min_length=1)
    highest_revenue: str
    currency: Literal["ARS"]

    @field_validator("total_revenue", "highest_revenue")
    @classmethod
    def validate_amount(cls, value: str) -> str:
        return ProductAnalysis.validate_amount(value)


class SalesAnalysisFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: Literal[False]
    errors: list[NonEmptyText] = Field(min_length=1)


sales_analysis_adapter = TypeAdapter(
    Annotated[
        SalesAnalysisSuccess | SalesAnalysisFailure,
        Field(discriminator="valid"),
    ]
)