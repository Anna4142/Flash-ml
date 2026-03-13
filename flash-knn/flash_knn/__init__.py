from flash_knn.interface import (
    FlashKNN,
    batch_knn_euclid,
    batch_knn_cosine,
)

try:
    from flash_knn.knn_triton import (
        batch_knn_euclid_triton,
        batch_knn_cosine_triton,
    )
except Exception:
    batch_knn_euclid_triton = None
    batch_knn_cosine_triton = None

__all__ = [
    "FlashKNN",
    "batch_knn_euclid",
    "batch_knn_cosine",
    "batch_knn_euclid_triton",
    "batch_knn_cosine_triton",
]
__version__ = "0.1.0"
