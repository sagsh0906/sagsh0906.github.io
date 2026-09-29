"""Readers/normalisers for instruction-tuning records (jsonl / json / parquet)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.suffix in {".jsonl", ".json", ".parquet"}:
                yield from iter_records(p)
        return
    if path.suffix == ".parquet":
        import pandas as pd

        for rec in pd.read_parquet(path).to_dict(orient="records"):
            yield _to_python(rec)
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        yield from (data if isinstance(data, list) else [data])
    else:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def _to_python(obj: Any) -> Any:
    """numpy arrays from parquet -> lists."""
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return [_to_python(x) for x in obj.tolist()]
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    return obj


_ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system", "tool": "tool"}


def to_messages(rec: dict[str, Any]) -> list[dict[str, str]] | None:
    """Normalise messages / ShareGPT / Alpaca / QA style records to chat messages."""
    if "messages" in rec and rec["messages"]:
        return [{"role": m["role"], "content": m["content"]} for m in rec["messages"]]
    if "conversations" in rec and rec["conversations"]:
        out = []
        for m in rec["conversations"]:
            role = _ROLE_MAP.get(m.get("from") or m.get("role"))
            if role:
                out.append({"role": role, "content": m.get("value") or m.get("content") or ""})
        return out
    if "instruction" in rec and ("output" in rec or "response" in rec):
        user = rec["instruction"] + (f"\n\n{rec['input']}" if rec.get("input") else "")
        return [{"role": "user", "content": user}, {"role": "assistant", "content": rec.get("output") or rec.get("response")}]
    if "question" in rec and "answer" in rec:
        return [{"role": "user", "content": rec["question"]}, {"role": "assistant", "content": rec["answer"]}]
    return None


def with_system(messages: list[dict[str, str]], system: str | None) -> list[dict[str, str]]:
    if not system:
        return messages
    if messages and messages[0]["role"] == "system":
        return messages
    return [{"role": "system", "content": system}] + messages


def write_jsonl(path: str | Path, rows) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n
