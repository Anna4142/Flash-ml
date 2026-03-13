# Flash-ml

Fast GPU ML algorithms (Triton / PyTorch). Each subdirectory is a self-contained package.

| Package      | Description |
|-------------|-------------|
| **[flash-knn](flash-knn/)** | Batched K-Nearest Neighbors with Triton (Euclidean & cosine). |

More algorithms (e.g. flash-kmeans, flash-pca) can be added as sibling directories later.

## Flash-KNN

```bash
cd flash-knn && pip install -e .
```

```python
import torch
from flash_knn import batch_knn_euclid

x = torch.randn(32, 10000, 128, device="cuda")
dist, idx = batch_knn_euclid(x, k=16)
```

See [flash-knn/README.md](flash-knn/README.md) for full docs.
