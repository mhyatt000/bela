from pathlib import Path

from . import common  # noqa: F401

try:
    from .batchspec import DEFAULT_SPEC, SIMPLE_SPEC, BatchSpec
except Exception:  # pragma: no cover - optional during tests
    DEFAULT_SPEC = SIMPLE_SPEC = BatchSpec = None

ROOT = Path(__file__).resolve().parent.parent

__all__ = ["BatchSpec", "DEFAULT_SPEC", "SIMPLE_SPEC"]
