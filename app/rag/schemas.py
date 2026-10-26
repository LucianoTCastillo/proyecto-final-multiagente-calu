from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchRequest(BaseModel):
    """Consulta que recibe la herramienta de búsqueda."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=10, strict=True)

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("La consulta no puede estar vacía.")

        return value


class DocumentChunk(BaseModel):
    """Fragmento de un documento con su fuente."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)


class SearchHit(DocumentChunk):
    """Fragmento recuperado y su puntuación de relevancia."""

    score: float = Field(ge=0, allow_inf_nan=False)


class SearchResponse(BaseModel):
    """Resultado validado de la búsqueda documental."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    hits: list[SearchHit] = Field(default_factory=list)