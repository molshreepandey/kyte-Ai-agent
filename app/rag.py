"""RAG component: chunk -> embed -> persist in a numpy vector store -> cosine search.

Embeddings come from OpenRouter. If the embeddings endpoint is unavailable the
store transparently falls back to a local hashing embedder, so retrieval still
works offline (lower quality, but deterministic and dependency-free).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI

from . import config

log = logging.getLogger("rag")

LOCAL_DIM = 768
INDEX_VERSION = 2   # bump when chunking / local embedding changes to invalidate cached indexes
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "is", "are", "of", "to", "and", "or", "in", "on", "for", "do",
    "i", "my", "me", "how", "what", "can", "does", "it", "at", "be", "with", "by",
}


@dataclass
class Chunk:
    text: str
    source: str
    title: str


# ---------------------------------------------------------------- chunking
def load_chunks(kb_dir: Path) -> list[Chunk]:
    """Split each markdown/text file on '## ' headings so each chunk is one topic."""
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        doc_title = path.stem
        for part in re.split(r"(?m)^(?=## )", text):
            part = part.strip()
            if not part:
                continue
            m = re.match(r"## (.+)", part)
            if m:
                title = m.group(1).strip()
            else:
                h1 = re.match(r"# (.+)", part)
                title = h1.group(1).strip() if h1 else doc_title
                if len(part.split()) < 25:   # bare document header, no content
                    doc_title = title
                    continue
            chunks.append(Chunk(text=part, source=path.name, title=title))
    return chunks


# -------------------------------------------------------------- embedders
def _stem(w: str) -> str:
    for suf in ("ing", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _grams(text: str) -> list[str]:
    toks = [_stem(w) for w in _TOKEN_RE.findall(text.lower()) if w not in _STOP]
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


def _fit_idf(texts: list[str]) -> dict[str, float]:
    df: dict[str, int] = {}
    for t in texts:
        for g in set(_grams(t)):
            df[g] = df.get(g, 0) + 1
    n = len(texts)
    return {g: math.log((1 + n) / (1 + c)) + 1 for g, c in df.items()}


def _local_embed(texts: list[str], idf: dict[str, float]) -> np.ndarray:
    """TF-IDF over unigrams+bigrams, hashed into a fixed dimension. No network."""
    out = np.zeros((len(texts), LOCAL_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for g in _grams(t):
            h = int.from_bytes(hashlib.md5(g.encode()).digest()[:8], "little")
            out[i, h % LOCAL_DIM] += idf.get(g, 1.0)
    return out


def _normalise(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1
    return m / n


class VectorStore:
    def __init__(self, kb_dir: Path = config.KB_DIR, data_dir: Path = config.DATA_DIR):
        self.kb_dir = kb_dir
        self.index_path = data_dir / "vector_store.npz"
        self.meta_path = data_dir / "vector_store.json"
        data_dir.mkdir(exist_ok=True)
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, 1), dtype=np.float32)
        self.embedder = "none"
        self._idf: dict[str, float] = {}
        self._client = (
            OpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL)
            if config.OPENROUTER_API_KEY
            else None
        )

    # ---- embedding backends
    def _remote_embed(self, texts: list[str]) -> np.ndarray:
        if self._client is None:
            raise RuntimeError("no API key")
        resp = self._client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
        data = sorted(resp.data, key=lambda d: d.index)
        return np.array([d.embedding for d in data], dtype=np.float32)

    def _embed(self, texts: list[str], backend: str) -> np.ndarray:
        if backend == "local":
            return _normalise(_local_embed(texts, self._idf))
        return _normalise(self._remote_embed(texts))

    # ---- build / load
    def _fingerprint(self, chunks: list[Chunk]) -> str:
        h = hashlib.sha256()
        for c in chunks:
            h.update(c.text.encode())
        return h.hexdigest()

    def build(self, force: bool = False) -> None:
        chunks = load_chunks(self.kb_dir)
        if not chunks:
            raise RuntimeError(f"No documents found in {self.kb_dir}")
        fp = self._fingerprint(chunks)
        self._idf = _fit_idf([f"{c.title}\n{c.text}" for c in chunks])

        if not force and self.index_path.exists() and self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            wanted = config.EMBEDDING_MODEL if self._client else "local"
            if meta.get("version") == INDEX_VERSION and meta["fingerprint"] == fp and meta["embedder"] in (wanted, "local"):
                self.matrix = np.load(self.index_path)["m"]
                self.chunks = chunks
                self.embedder = meta["embedder"]
                log.info("Loaded cached index (%d chunks, %s)", len(chunks), self.embedder)
                return

        texts = [f"{c.title}\n{c.text}" for c in chunks]
        backend = "remote"
        try:
            matrix = self._embed(texts, "remote")
            self.embedder = config.EMBEDDING_MODEL
        except Exception as exc:  # no key / endpoint down / model unsupported
            log.warning("Remote embeddings unavailable (%s); using local embedder", exc)
            matrix = self._embed(texts, "local")
            self.embedder, backend = "local", "local"

        self.chunks, self.matrix = chunks, matrix
        np.savez_compressed(self.index_path, m=matrix)
        self.meta_path.write_text(
            json.dumps({"fingerprint": fp, "embedder": self.embedder, "version": INDEX_VERSION}), encoding="utf-8"
        )
        log.info("Built index: %d chunks via %s", len(chunks), backend)

    # ---- query
    def search(self, query: str, k: int = 3) -> list[dict]:
        if not self.chunks:
            self.build()
        backend = "local" if self.embedder == "local" else "remote"
        try:
            q = self._embed([query], backend)
        except Exception as exc:
            log.warning("Query embedding failed (%s); rebuilding with local embedder", exc)
            texts = [f"{c.title}\n{c.text}" for c in self.chunks]
            self.matrix, self.embedder = self._embed(texts, "local"), "local"
            q = self._embed([query], "local")
        scores = (self.matrix @ q[0]).tolist()
        order = np.argsort(scores)[::-1][:k]
        return [
            {
                "score": round(float(scores[i]), 4),
                "source": self.chunks[i].source,
                "title": self.chunks[i].title,
                "text": self.chunks[i].text,
            }
            for i in order
        ]
