"""Load HGB v2 across numpy BitGenerator pickle skew; rewrite as v3."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import numpy.random._pickle as nrp
from joblib.numpy_pickle import NumpyUnpickler
from numpy.random import PCG64

_ORIG = nrp.__bit_generator_ctor


class _TolerantPCG64(PCG64):
    def __setstate__(self, state):
        try:
            return super().__setstate__(state)
        except Exception:
            super().__init__(42)
            return None


def _ctor(bit_generator_name="PCG64"):
    # Pickle may pass a class, an instance, or a name string.
    if isinstance(bit_generator_name, (_TolerantPCG64, PCG64)):
        return bit_generator_name
    if isinstance(bit_generator_name, type):
        try:
            return bit_generator_name()
        except Exception:
            bit_generator_name = bit_generator_name.__name__
    if not isinstance(bit_generator_name, str):
        if hasattr(bit_generator_name, "random_raw"):
            return bit_generator_name
    # Always return a tolerant instance so mismatched state formats are ignored.
    return _TolerantPCG64(42)


nrp.__bit_generator_ctor = _ctor
# Defaults capture the ctor at def-time — refresh them.
nrp.__generator_ctor.__defaults__ = ("MT19937", _ctor)
nrp.__randomstate_ctor.__defaults__ = ("MT19937", _ctor)


class SoftNumpyUnpickler(NumpyUnpickler):
    def find_class(self, module, name):
        if module.startswith("numpy.random") and name == "PCG64":
            return _TolerantPCG64
        return super().find_class(module, name)


def load_model(path: Path):
    with open(path, "rb") as f:
        return SoftNumpyUnpickler(str(path), f, mmap_mode=None).load()


def main() -> None:
    src = Path("artifacts/rank_model_hgb_v2.joblib")
    dst = Path("artifacts/rank_model_hgb_v3.joblib")
    blob = load_model(src)
    print("loaded keys", list(blob.keys()), flush=True)
    model = blob["model"]
    X = np.zeros((3, 30), dtype=np.float32)
    print("proba", model.predict_proba(X)[:, 1], flush=True)
    blob = dict(blob)
    blob["sibling_accept"] = True
    blob["reloaded_for"] = "numpy-compat + raise-score plan"
    joblib.dump(blob, dst)
    nrp.__bit_generator_ctor = _ORIG
    nrp.__generator_ctor.__defaults__ = ("MT19937", _ORIG)
    nrp.__randomstate_ctor.__defaults__ = ("MT19937", _ORIG)
    blob2 = joblib.load(dst)
    print("reload_ok", blob2["model"].predict_proba(X)[:, 1], flush=True)
    print("rule", blob2.get("rule"), flush=True)
    print("WROTE", dst, flush=True)


if __name__ == "__main__":
    main()
