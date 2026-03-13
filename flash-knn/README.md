# Flash-KNN

Fast batched K-Nearest Neighbors with Triton GPU kernels, in the same style as [flash-kmeans](https://github.com/svg-project/flash-kmeans).

- **Batched**: process `(B, N, D)` queries and `(B, M, D)` references in one go.
- **Triton**: tiled distance computation (Euclidean and cosine) with chunked refs and top-K merge.
- **Fallback**: PyTorch (cdist / bmm) when Triton is unavailable.

## Installation

```bash
cd flash-knn
pip install -e .
```

## Usage

```python
import torch
from flash_knn import batch_knn_euclid, batch_knn_cosine, FlashKNN

# (B, N, D) queries, (B, M, D) refs
x = torch.randn(32, 10000, 128, device="cuda", dtype=torch.float16)
ref = torch.randn(32, 50000, 128, device="cuda", dtype=torch.float16)

# Squared L2 distance, k=16
dist, idx = batch_knn_euclid(x, ref=ref, k=16)
# dist: (32, 10000, 16), idx: (32, 10000, 16)

# Cosine (normalize first)
x_n = torch.nn.functional.normalize(x, p=2, dim=-1)
ref_n = torch.nn.functional.normalize(ref, p=2, dim=-1)
sim, idx = batch_knn_cosine(x_n, ref=ref_n, k=16)

# Class API
knn = FlashKNN(k=16, metric="euclidean")
knn.fit(ref)
dist, idx = knn.kneighbors(x)
```

## Design

- Same tiling idea as flash-kmeans assign: tile over query dimension `N`, iterate over reference dimension `M` in chunks (`BLOCK_M`).
- Each Triton kernel computes a `(BLOCK_N, BLOCK_M)` block of distances/similarities; Python loops over ref chunks and merges with `torch.topk` so full `(B, N, M)` is never materialized.
- Supports self-KNN by passing `ref=None` (uses `x` as reference).
