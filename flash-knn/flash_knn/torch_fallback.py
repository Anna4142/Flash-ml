"""PyTorch fallback for batched KNN when Triton is unavailable or for CPU."""
from typing import Optional, Tuple
import torch


def batch_knn_euclid_torch(
    x: torch.Tensor,
    ref: Optional[torch.Tensor] = None,
    k: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched K-NN with squared L2 using cdist + topk. O(B*N*M) memory."""
    if ref is None:
        ref = x
    B, N, D = x.shape
    M = ref.shape[1]
    k = min(k, M)
    # (B, N, M)
    dist = torch.cdist(x.float(), ref.float(), p=2).pow(2)
    best_dist, best_idx = dist.topk(k, dim=-1, largest=False)
    return best_dist, best_idx


def batch_knn_cosine_torch(
    x: torch.Tensor,
    ref: Optional[torch.Tensor] = None,
    k: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched K-NN with cosine similarity. x and ref should be L2-normalized."""
    if ref is None:
        ref = x
    B, N, D = x.shape
    M = ref.shape[1]
    k = min(k, M)
    sim = torch.bmm(x.float(), ref.float().transpose(1, 2))
    best_sim, best_idx = sim.topk(k, dim=-1, largest=True)
    return best_sim, best_idx
