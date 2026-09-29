import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def fresh_store(tmp_path):
    """Isolate the artifact store per test."""
    from matbrain.mcp.artifacts import ArtifactStore, set_store

    set_store(ArtifactStore())
    yield


@pytest.fixture
def nacl():
    from pymatgen.core import Lattice, Structure

    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


@pytest.fixture
def registry():
    from matbrain.mcp import load_registry

    return load_registry()
