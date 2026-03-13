"""
Batched KNN orchestration: imports assign kernels from assign_euclid_triton and
assign_cosine_triton, loops over ref chunks, merges with topk. Same pattern as
flash-kmeans kmeans_triton_impl (assign + loop + merge).
"""
from typing import Optional, Tuple
import torch
import triton

from flash_knn.assign_euclid_triton import knn_euclid_dist_chunk_triton
from flash_knn.assign_cosine_triton import knn_cosine_sim_chunk_triton


def batch_knn_euclid_triton(
    x: torch.Tensor,
    ref: Optional[torch.Tensor] = None,
    k: int = 1,
    *,
    x_sq: Optional[torch.Tensor] = None,
    ref_sq: Optional[torch.Tensor] = None,
    BLOCK_N: int = 128,
    BLOCK_M: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched K-NN with squared Euclidean distance.
    Uses assign_euclid_triton for distance chunks, then topk merge (like kmeans assign + update).
    """
    if ref is None:
        ref = x
    assert x.is_cuda and ref.is_cuda
    B, N, D = x.shape
    M = ref.shape[1]
    assert ref.shape == (B, M, D)
    k = min(k, M)

    if x_sq is None:
        x_sq = (x.to(torch.float32) ** 2).sum(dim=-1)
    if ref_sq is None:
        ref_sq = (ref.to(torch.float32) ** 2).sum(dim=-1)
    x = x.contiguous()
    ref = ref.contiguous()

    best_dist = torch.full((B, N, k), 3.4e38, device=x.device, dtype=torch.float32)
    best_idx = torch.full((B, N, k), -1, device=x.device, dtype=torch.int64)

    chunk_buf = torch.empty((B, N, BLOCK_M), device=x.device, dtype=torch.float32)

    for m_start in range(0, M, BLOCK_M):
        m_end = min(m_start + BLOCK_M, M)
        ref_chunk = ref[:, m_start:m_end, :]
        ref_sq_chunk = ref_sq[:, m_start:m_end]
        knn_euclid_dist_chunk_triton(
            x,
            ref_chunk,
            x_sq,
            ref_sq_chunk,
            chunk_buf[:, :, : m_end - m_start],
            BLOCK_N=BLOCK_N,
            BLOCK_M=BLOCK_M,
        )
        dist_chunk = chunk_buf[:, :, : m_end - m_start]
        idx_chunk = m_start + torch.arange(m_end - m_start, device=x.device, dtype=torch.int64)
        idx_chunk = idx_chunk.unsqueeze(0).unsqueeze(0).expand(B, N, -1)
        combined_dist = torch.cat([best_dist, dist_chunk], dim=-1)
        combined_idx = torch.cat([best_idx, idx_chunk], dim=-1)
        best_dist, topk_i = combined_dist.topk(k, dim=-1, largest=False)
        best_idx = combined_idx.gather(-1, topk_i)

    return best_dist, best_idx


def batch_knn_cosine_triton(
    x: torch.Tensor,
    ref: Optional[torch.Tensor] = None,
    k: int = 1,
    *,
    BLOCK_N: int = 128,
    BLOCK_M: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched K-NN with cosine similarity (larger = nearer).
    Uses assign_cosine_triton for similarity chunks, then topk merge.
    """
    if ref is None:
        ref = x
    assert x.is_cuda and ref.is_cuda
    B, N, D = x.shape
    M = ref.shape[1]
    assert ref.shape == (B, M, D)
    k = min(k, M)

    x = x.contiguous()
    ref = ref.contiguous()
    best_sim = torch.full((B, N, k), -3.4e38, device=x.device, dtype=torch.float32)
    best_idx = torch.full((B, N, k), -1, device=x.device, dtype=torch.int64)
    chunk_buf = torch.empty((B, N, BLOCK_M), device=x.device, dtype=torch.float32)

    for m_start in range(0, M, BLOCK_M):
        m_end = min(m_start + BLOCK_M, M)
        ref_chunk = ref[:, m_start:m_end, :]
        knn_cosine_sim_chunk_triton(
            x,
            ref_chunk,
            chunk_buf[:, :, : m_end - m_start],
            BLOCK_N=BLOCK_N,
            BLOCK_M=BLOCK_M,
        )
        sim_chunk = chunk_buf[:, :, : m_end - m_start]
        idx_chunk = m_start + torch.arange(m_end - m_start, device=x.device, dtype=torch.int64)
        idx_chunk = idx_chunk.unsqueeze(0).unsqueeze(0).expand(B, N, -1)
        combined_sim = torch.cat([best_sim, sim_chunk], dim=-1)
        combined_idx = torch.cat([best_idx, idx_chunk], dim=-1)
        best_sim, topk_i = combined_sim.topk(k, dim=-1, largest=True)
        best_idx = combined_idx.gather(-1, topk_i)

    return best_sim, best_idx
