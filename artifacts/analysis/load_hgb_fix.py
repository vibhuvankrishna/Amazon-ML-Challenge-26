"""Try to load rank_model_hgb_v2 despite numpy BitGenerator pickle skew."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from numpy.random.bit_generator import BitGenerator
import numpy.random._pickle as nrp

_orig_setstate = BitGenerator.__setstate__
_orig_ctor = nrp.__bit_generator_ctor


def _safe_setstate(self, state):
    try:
        return _orig_setstate(self, state)
    except Exception as e:
        print("setstate fail", type(e).__name__, e, "state_type", type(state))
        return None


def _ctor(bit_generator_name="PCG64"):
    print("ctor", bit_generator_name, type(bit_generator_name))
    if isinstance(bit_generator_name, type):
        return bit_generator_name()
    if not isinstance(bit_generator_name, str):
        name = getattr(bit_generator_name, "__name__", None) or "PCG64"
        return _orig_ctor(name)
    return _orig_ctor(bit_generator_name)


BitGenerator.__setstate__ = _safe_setstate
nrp.__bit_generator_ctor = _ctor

path = Path("artifacts/rank_model_hgb_v2.joblib")
b = joblib.load(path)
print("SUCCESS", b.keys())
m = b["model"]
X = np.zeros((2, 30), dtype=np.float32)
print("proba", m.predict_proba(X)[:, 1])
out = Path("artifacts/rank_model_hgb_v3.joblib")
joblib.dump(b, out)
print("saved", out)
