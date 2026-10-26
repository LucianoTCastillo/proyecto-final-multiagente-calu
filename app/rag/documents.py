import asyncio
import hashlib
import re
from pathlib import Path

from app.rag.schemas import DocumentChunk


# Raíz del proyecto, independientemente de dónde se ejecute.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def split_text(
    text: str,
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[str]:
    """Divide texto por caracteres, manteniendo un solapamiento."""
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor que cero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap debe ser mayor o igual a cero "
            "y menor que chunk_size."
        )

    text = text.strip()

    if not text:
        return []

    fragments = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Intenta cortar entre palabras.
        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start + chunk_size // 2:
                end = boundary

        fragment = text[start:end].strip()

        if fragment:
            fragments.append(fragment)

        if end == len(text):
            break

        start = max(start + 1, end - overlap)

    return fragments


def _load_documents(
    documents_dir: Path,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    """Lee los archivos Markdown y valida sus fragmentos."""
    if not documents_dir.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta de documentos: {documents_dir}"
        )

    paths = sorted(documents_dir.rglob("*.md"))

    if not paths:
        raise ValueError(
            "No se encontraron documentos Markdown."
        )

    chunks = []

    for path in paths:
        text = path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        heading = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = heading.group(1).strip() if heading else path.stem

        # Guarda una ruta relativa, sin exponer rutas de la computadora.
        source = path.relative_to(documents_dir).as_posix()

        # Separa el documento por encabezados Markdown.
        sections = re.split(
            r"(?m)(?=^#{1,6}\s)",
            text,
        )

        fragments = []

        for section in sections:
            section = section.strip()

            if not section:
                continue

            if len(section) <= chunk_size:
                fragments.append(section)
            else:
                fragments.extend(
                    split_text(section, chunk_size, overlap)
                )

        for position, content in enumerate(fragments):
            identifier = hashlib.sha256(
                f"{source}:{position}:{content}".encode("utf-8")
            ).hexdigest()[:20]

            chunks.append(
                DocumentChunk(
                    chunk_id=identifier,
                    source=source,
                    title=title,
                    content=content,
                )
            )

    if not chunks:
        raise ValueError(
            "Los documentos no contienen texto para procesar."
        )

    return chunks


async def load_documents(
    documents_dir: Path | None = None,
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[DocumentChunk]:
    """Carga documentos sin bloquear el event loop."""
    directory = (
        Path(documents_dir)
        if documents_dir is not None
        else DEFAULT_DOCUMENTS_DIR
    )

    return await asyncio.to_thread(
        _load_documents,
        directory,
        chunk_size,
        overlap,
    )