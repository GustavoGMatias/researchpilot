import hashlib
import json
import logging
import math
from pathlib import Path

log = logging.getLogger(__name__)

ALLOWED = {".txt", ".md"}
CHUNK_SIZE = 800
OVERLAP = 100
BATCH = 50
MIN_SCORE = 0.65


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class DocumentIndex:
    def __init__(self, llm, docs_dir: str, index_path: str):
        self.llm = llm
        self.docs_dir = Path(docs_dir)
        self.index_path = Path(index_path)
        self.entries: list[dict] = []
        if self.index_path.exists():
            try:
                self.entries = json.loads(self.index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                log.exception("índice corrompido; será reconstruído")

    def sync(self) -> int:
        """Reindexa só os arquivos novos/alterados. Retorna nº de arquivos reindexados."""
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        files = {p.name: p for p in self.docs_dir.iterdir() if p.suffix.lower() in ALLOWED}
        keep = []
        for e in self.entries:
            p = files.get(e["source"])
            if p and _hash(p) == e["hash"]:
                keep.append(e)
        done = {e["source"] for e in keep}
        new_entries, changed = [], 0
        for name, p in files.items():
            if name in done:
                continue
            h = _hash(p)
            chunks = chunk_text(p.read_text(encoding="utf-8", errors="ignore"))
            if not chunks:
                continue
            vectors = []
            for i in range(0, len(chunks), BATCH):
                vectors += self.llm.embed(chunks[i : i + BATCH], "RETRIEVAL_DOCUMENT")
            new_entries += [
                {"source": name, "hash": h, "text": c, "embedding": v}
                for c, v in zip(chunks, vectors)
            ]
            changed += 1
        if changed or len(keep) != len(self.entries):
            self.entries = keep + new_entries
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.index_path.write_text(json.dumps(self.entries), encoding="utf-8")
        log.info("índice: %d trechos, %d arquivos reindexados", len(self.entries), changed)
        return changed

    def search(self, query: str, k: int = 4) -> dict:
        self.sync()
        if not self.entries:
            return {"results": [], "sources": [], "note": "Nenhum documento indexado em docs/."}
        q = self.llm.embed([query], "RETRIEVAL_QUERY")[0]
        scored = sorted(
            ((cosine(q, e["embedding"]), e) for e in self.entries),
            key=lambda t: t[0],
            reverse=True,
        )[:k]
        results = [
            {"source": e["source"], "score": round(s, 3), "text": e["text"]}
            for s, e in scored
            if s >= MIN_SCORE
        ]
        if not results:
            return {"results": [], "sources": [], "note": "Nenhum trecho relevante nos documentos."}
        sources = sorted({f"docs/{r['source']}" for r in results})
        return {"results": results, "sources": sources}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
