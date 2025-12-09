from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class DocumentChunk:
    doc_id: str
    title: str
    text: str


DEFAULT_DOCS = [
    DocumentChunk(
        doc_id="project_overview",
        title="Project Overview",
        text=(
            "Nebula SIWI Bot is a ChatBot and lightweight RAG service. "
            "It ships with a Python backend and a modern frontend, and works out of the box without NebulaGraph. "
            "Graph experiments remain available as an optional module."
        ),
    ),
    DocumentChunk(
        doc_id="usage",
        title="Usage",
        text=(
            "Run the backend, then call POST /api/chat with a message to receive an answer with sources. "
            "RAG uses local demo documents by default. "
            "You can add more files under data/demo_docs to change the retrieval set."
        ),
    ),
]


class DocLoader:
    """Load demo documents from disk and split into lightweight chunks."""

    def __init__(self, data_dir: Path, chunk_size: int = 500):
        self.data_dir = data_dir
        self.chunk_size = chunk_size

    def load(self) -> List[DocumentChunk]:
        if not self.data_dir.exists():
            return DEFAULT_DOCS

        documents: List[DocumentChunk] = []
        for path in sorted(self.data_dir.glob("**/*")):
            if path.is_dir():
                continue
            if path.suffix.lower() not in {".txt", ".md"}:
                continue
            documents.extend(self._load_file(path))

        return documents or DEFAULT_DOCS

    def _load_file(self, path: Path) -> Iterable[DocumentChunk]:
        text = path.read_text(encoding="utf-8")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        title = lines[0] if lines else path.stem.replace("_", " ").title()

        for idx, chunk in enumerate(self._split_text(text)):
            doc_id = f"{path.stem}-{idx}"
            yield DocumentChunk(doc_id=doc_id, title=title, text=chunk)

    def _split_text(self, text: str) -> List[str]:
        parts: List[str] = []
        current: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                if current:
                    parts.append(" ".join(current))
                    current = []
                continue
            current.append(stripped)
            if sum(len(x) for x in current) >= self.chunk_size:
                parts.append(" ".join(current))
                current = []

        if current:
            parts.append(" ".join(current))

        return parts
