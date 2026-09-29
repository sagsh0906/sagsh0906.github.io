"""General information retrieval tools: web search, literature search (SearXNG)
and local knowledge retrieval over an internal literature repository."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import httpx
from pydantic import Field

from matbrain.mcp.registry import REGISTRY, SEARCH, ToolArgs, ToolError


class SearchArgs(ToolArgs):
    query: str = Field(..., min_length=2, max_length=500)
    top_k: int = Field(5, ge=1, le=20)


@REGISTRY.tool(category=SEARCH)
def web_search(args: SearchArgs) -> dict:
    """Real-time web search. Uses the Bing Web Search API when BING_API_KEY is set
    (as in the paper); otherwise falls back to the configured SearXNG instance."""
    key = os.environ.get("BING_API_KEY")
    if key:
        endpoint = os.environ.get("BING_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")
        r = httpx.get(endpoint, params={"q": args.query, "count": args.top_k}, headers={"Ocp-Apim-Subscription-Key": key}, timeout=30)
        r.raise_for_status()
        pages = r.json().get("webPages", {}).get("value", [])
        return {"engine": "bing", "results": [{"title": p.get("name"), "url": p.get("url"), "snippet": p.get("snippet")} for p in pages[: args.top_k]]}
    return _searxng(args.query, args.top_k, categories="general")


def _searxng(query: str, top_k: int, categories: str) -> dict:
    base = os.environ.get("SEARXNG_URL")
    if not base:
        raise ToolError("no search backend configured (set BING_API_KEY or SEARXNG_URL)")
    r = httpx.get(f"{base.rstrip('/')}/search", params={"q": query, "format": "json", "categories": categories}, timeout=30)
    r.raise_for_status()
    rows = r.json().get("results", [])[:top_k]
    return {"engine": "searxng", "results": [{"title": x.get("title"), "url": x.get("url"), "snippet": x.get("content"), "published": x.get("publishedDate")} for x in rows]}


@REGISTRY.tool(category=SEARCH)
def literature_search(args: SearchArgs) -> dict:
    """Scientific literature search through SearXNG (science category: arXiv,
    Crossref, Semantic Scholar, Google Scholar ...)."""
    return _searxng(args.query, args.top_k, categories="science")


# --------------------------------------------------------------------------- #
# Local knowledge retrieval (BM25 over markdown chunks, e.g. MinerU output)
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"[A-Za-z][a-z]?\d*(?:\.\d+)?|[a-z]+|\d+(?:\.\d+)?", re.I)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class BM25Index:
    def __init__(self, chunks: list[tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(_tokens(text)) for _, text in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lengths) / max(len(self.docs), 1)
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, str, str]]:
        q = _tokens(query)
        scores = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                if t in d:
                    tf = d[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl))
            if s > 0:
                scores.append((s, *self.chunks[i]))
        return sorted(scores, key=lambda x: -x[0])[:top_k]


def chunk_markdown(text: str, max_chars: int = 2000) -> list[str]:
    """Split markdown on headings, then on paragraphs to respect max_chars."""
    sections = re.split(r"\n(?=#{1,6} )", text)
    out = []
    for sec in sections:
        buf = ""
        for para in sec.split("\n\n"):
            if len(buf) + len(para) > max_chars and buf:
                out.append(buf.strip())
                buf = ""
            buf += para + "\n\n"
        if buf.strip():
            out.append(buf.strip())
    return out


@lru_cache(maxsize=2)
def _kb_index(root: str) -> BM25Index:
    chunks = []
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            for i, c in enumerate(chunk_markdown(path.read_text(encoding="utf-8", errors="ignore"))):
                chunks.append((f"{path.name}#{i}", c))
    if not chunks:
        raise ToolError(f"knowledge base at {root} is empty")
    return BM25Index(chunks)


@REGISTRY.tool(category=SEARCH)
def local_knowledge_search(args: SearchArgs) -> dict:
    """Retrieve passages from the internal literature repository (MATBRAIN_KB_DIR)."""
    root = os.environ.get("MATBRAIN_KB_DIR")
    if not root:
        raise ToolError("MATBRAIN_KB_DIR is not configured")
    hits = _kb_index(root).search(args.query, args.top_k)
    return {"results": [{"source": src, "score": round(score, 3), "text": text[:1200]} for score, src, text in hits]}
