"""
Shared fixtures. Importantly: redirects every sqlite DB module to a
throwaway tmp-path file before any test runs, so the test suite never reads
or writes backend/{history,contacts,family}.db -- the real local data used
by the running app/extension.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import contacts_db  # noqa: E402
import family_db  # noqa: E402
import history_db  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_dbs(tmp_path, monkeypatch):
    monkeypatch.setattr(history_db, "DB_PATH", tmp_path / "history.db")
    monkeypatch.setattr(contacts_db, "DB_PATH", tmp_path / "contacts.db")
    monkeypatch.setattr(family_db, "DB_PATH", tmp_path / "family.db")
    history_db.init_db()
    contacts_db.init_db()
    family_db.init_db()
    yield


@pytest.fixture
def client(isolated_dbs):
    # Explicit dependency on isolated_dbs (rather than relying on autouse
    # ordering) guarantees DB_PATH is patched before main's module-level
    # init_db() calls run on first import.
    from fastapi.testclient import TestClient

    import main

    return TestClient(main.app)


def textured_image(size=(256, 256), seed=0) -> Image.Image:
    """A sharp, high-frequency-content image -- passes the blur/sharpness
    quality gate (preprocessing.assess_face_crop_quality), unlike a flat
    solid color (sharpness 0, always fails it). Stands in for "a real,
    in-focus photo" in tests that aren't specifically exercising the
    blur-rejection path."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(array)


def jpeg_bytes(size=(256, 256), color=None, seed=0) -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGB", size, color) if color is not None else textured_image(size, seed=seed)
    image.save(buffer, format="JPEG")
    return buffer.getvalue()
