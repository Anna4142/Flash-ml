"""
Batched KNN Triton entrypoint: re-exports from assign modules + knn_triton_impl.
Same layout as flash-kmeans: assign_euclid_triton, assign_cosine_triton,
and knn_triton_impl that calls them.
"""
from flash_knn.knn_triton_impl import (
    batch_knn_euclid_triton,
    batch_knn_cosine_triton,
)
from flash_knn.assign_euclid_triton import knn_euclid_dist_chunk_triton
from flash_knn.assign_cosine_triton import knn_cosine_sim_chunk_triton

__all__ = [
    "batch_knn_euclid_triton",
    "batch_knn_cosine_triton",
    "knn_euclid_dist_chunk_triton",
    "knn_cosine_sim_chunk_triton",
]
