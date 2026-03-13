# Flash-ml

Fast GPU ML algorithms (Triton / PyTorch). Each subdirectory is a self-contained package.

| Package      | Description |
|-------------|-------------|
| **[flash-knn](flash-knn/)** | Batched K-Nearest Neighbors with Triton (Euclidean & cosine). |

Useful for semantic deduplication (e.g. SemDeDup), cluster-based sampling / diversification for LLM training (e.g. ClusterClip), and other embedding-based pipelines. More algorithms can be added as sibling directories later.

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

### Benchmark results (H200, L2, fp16)

flash-knn matches PyTorch numerically (correctness checked). Sample timings from `benchmarks/benchmark_knn_libs.py` (L2 sweep, k=1):

| B | Q   | N    | flash-knn (ms) | PyTorch (ms) | speedup (torch/repo) |
|---|-----|------|-----------------|--------------|----------------------|
| 1 | 256 | 4096 | 2.92            | 0.19         | 0.064                |
| 1 | 512 | 8192 | 5.76            | 0.22         | 0.038                |
| 1 | 1024 | 16384 | 11.52         | 0.57         | 0.049                |
| 1 | 2048 | 32768 | 22.86         | 1.81         | 0.079                |
| 2 | 1024 | 16384 | 11.62         | 0.98         | 0.085                |
| 4 | 1024 | 16384 | 11.95         | 1.83         | 0.15                 |

Run benchmarks: `cd flash-knn && pip install -e . && ./benchmarks/run_all.sh`. Results in `flash-knn/benchmarks/results_knn_libs.csv`.
