from importlib import import_module
from pathlib import Path

from . import common  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent

__all__ = ["BatchSpec", "DEFAULT_SPEC", "SIMPLE_SPEC"]


def __getattr__(name):
    if name in __all__:
        mod = import_module(".batchspec", __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(name)
