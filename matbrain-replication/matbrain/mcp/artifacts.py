"""CIF artefact store.

Tools exchange crystal structures through short handles (``cif://<id>``)
instead of pasting full CIF text into the context window. This is the
"CIF artefact-handle transfer" that the paper credits for Mat-T1's more stable
multistep structure-design execution, and it keeps observations within the
tool-response budget used during RL (max_tool_response_length).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pymatgen.core import Structure

from matbrain.chem import parse_structure, structure_to_cif

HANDLE_PREFIX = "cif://"
HANDLE_RE = re.compile(r"^cif://[0-9a-f]{12}$")


def is_handle(text: str) -> bool:
    return isinstance(text, str) and bool(HANDLE_RE.match(text.strip()))


@dataclass
class Artifact:
    handle: str
    cif: str
    meta: dict[str, Any] = field(default_factory=dict)


class ArtifactStore:
    def __init__(self, root: str | os.PathLike | None = None):
        self._items: dict[str, Artifact] = {}
        self._lock = threading.Lock()
        self.root = Path(root) if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self._items)

    def put_cif(self, cif: str, meta: dict[str, Any] | None = None) -> str:
        handle = HANDLE_PREFIX + hashlib.sha1(cif.encode()).hexdigest()[:12]
        art = Artifact(handle, cif, dict(meta or {}))
        with self._lock:
            self._items[handle] = art
        if self.root:
            stem = handle.removeprefix(HANDLE_PREFIX)
            (self.root / f"{stem}.cif").write_text(cif, encoding="utf-8")
            (self.root / f"{stem}.json").write_text(json.dumps(art.meta, default=str), encoding="utf-8")
        return handle

    def put_structure(self, structure: Structure, meta: dict[str, Any] | None = None) -> str:
        meta = dict(meta or {})
        meta.setdefault("formula", structure.composition.reduced_formula)
        meta.setdefault("num_sites", len(structure))
        return self.put_cif(structure_to_cif(structure), meta)

    def exists(self, handle: str) -> bool:
        return self._load(handle) is not None

    def _load(self, handle: str) -> Artifact | None:
        handle = handle.strip()
        with self._lock:
            art = self._items.get(handle)
        if art is None and self.root:
            stem = handle.removeprefix(HANDLE_PREFIX)
            path = self.root / f"{stem}.cif"
            if path.exists():
                meta_path = self.root / f"{stem}.json"
                meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                art = Artifact(handle, path.read_text(encoding="utf-8"), meta)
                with self._lock:
                    self._items[handle] = art
        return art

    def get(self, handle: str) -> Artifact:
        art = self._load(handle)
        if art is None:
            raise KeyError(f"unknown artifact handle {handle!r}")
        return art

    def get_cif(self, handle: str) -> str:
        return self.get(handle).cif

    def resolve_cif(self, cif_or_handle: str) -> str:
        return self.get_cif(cif_or_handle) if is_handle(cif_or_handle) else cif_or_handle

    def resolve_structure(self, cif_or_handle: str | Structure) -> Structure:
        if isinstance(cif_or_handle, Structure):
            return cif_or_handle
        return parse_structure(self.resolve_cif(cif_or_handle))


_STORE: ArtifactStore | None = None


def get_store() -> ArtifactStore:
    global _STORE
    if _STORE is None:
        _STORE = ArtifactStore(os.environ.get("MATBRAIN_ARTIFACT_DIR"))
    return _STORE


def set_store(store: ArtifactStore) -> None:
    global _STORE
    _STORE = store
