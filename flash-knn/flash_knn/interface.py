"""Sklearn-like and high-level API for flash-knn."""
from __future__ import annotations
from typing import Optional, Tuple
import torch

try:
    from flash_knn.knn_triton import (
        batch_knn_euclid_triton,
        batch_knn_cosine_triton,
    )
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

from flash_knn.torch_fallback import (
    batch_knn_euclid_torch,
    batch_knn_cosine_torch,
)


def _require_cuda(x: torch.Tensor) -> None:
    if not x.is_cuda:
        raise RuntimeError("flash_knn Triton API requires CUDA tensors.")


class FlashKNN:
    """
    Fast batched K-Nearest Neighbors (same style as FlashKMeans).

    Parameters
    ----------
    k : int
        Number of neighbors.
    metric : str
        "euclidean" (squared L2) or "cosine".
    use_triton : bool
        Use Triton kernels when on CUDA; otherwise PyTorch fallback.
    """

    def __init__(
        self,
        k: int = 1,
        metric: str = "euclidean",
        use_triton: bool = True,
    ):
        self.k = int(k)
        self.metric = str(metric).lower()
        self.use_triton = bool(use_triton) and _HAS_TRITON
        if self.metric not in ("euclidean", "cosine"):
            raise ValueError("metric must be 'euclidean' or 'cosine'")

    def fit(self, x: torch.Tensor) -> "FlashKNN":
        """Store reference set (for fit + query pattern). Not required for batch_knn_*."""
        self.ref_ = x
        return self

    def kneighbors(
        self,
        x: torch.Tensor,
        ref: Optional[torch.Tensor] = None,
        k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return K nearest neighbors for each query.

        x: (B, N, D) queries.
        ref: (B, M, D) references; if None, use self.ref_ or x (self-KNN).
        k: override self.k if provided.

        Returns
        -------
        dist_or_sim : (B, N, k)
            Distances (euclidean) or similarities (cosine).
        idx : (B, N, k)
            Reference indices (int64).
        """
        k = k if k is not None else self.k
        ref = ref if ref is not None else getattr(self, "ref_", x)
        if x.is_cuda and self.use_triton:
            if self.metric == "euclidean":
                return batch_knn_euclid_triton(x, ref=ref, k=k)
            return batch_knn_cosine_triton(x, ref=ref, k=k)
        if self.metric == "euclidean":
            return batch_knn_euclid_torch(x, ref=ref, k=k)
        return batch_knn_cosine_torch(x, ref=ref, k=k)


def batch_knn_euclid(
    x: torch.Tensor,
    ref: Optional[torch.Tensor] = None,
    k: int = 1,
    use_triton: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched K-NN with squared Euclidean distance."""
    if x.is_cuda and use_triton and _HAS_TRITON:
        return batch_knn_euclid_triton(x, ref=ref, k=k)
    return batch_knn_euclid_torch(x, ref=ref, k=k)


def batch_knn_cosine(
    x: torch.Tensor,
    ref: Optional[torch.Tensor] = None,
    k: int = 1,
    use_triton: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched K-NN with cosine similarity (x/ref L2-normalized)."""
    if x.is_cuda and use_triton and _HAS_TRITON:
        return batch_knn_cosine_triton(x, ref=ref, k=k)
    return batch_knn_cosine_torch(x, ref=ref, k=k)
