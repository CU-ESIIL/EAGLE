import os

try:
    from .pointrope_cuda import PointROPE
except Exception as e:  # EAGLE patch: CUDA kernel is optional (see pointrope/setup.py)
    if os.environ.get("EAGLE_VERBOSE_POINTROPE"):
        print(
            f"[PointROPE] CUDA implementation unavailable ({type(e).__name__}: {e}). "
            "Using slower Pytorch fallback."
        )
    from .pointrope_torch import PointROPE

__all__ = ["PointROPE"]
