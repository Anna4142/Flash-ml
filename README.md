# Flash-ml

Fast GPU ML algorithms (Triton / PyTorch). Each subdirectory is a self-contained package.

### Clone and run (on a GPU machine with CUDA)

```bash
git clone https://github.com/Anna4142/Flash-ml.git
cd Flash-ml/flash-knn
pip install -e .
bash benchmarks/run_all.sh
```

Results in `flash-knn/benchmarks/results_knn.jsonl`. Requires Linux + CUDA (Triton is not available on macOS).

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

### Benchmark results (H200, L2, fp16, k=1)

flash-knn matches PyTorch numerically (correctness checked) and is **faster** than the PyTorch baseline on H200:

| B | Q    | N     | flash-knn (ms) | PyTorch (ms) |
|---|------|-------|----------------|--------------|
| 1 | 1024 | 8192  | **0.28**       | 0.30         |
| 1 | 1024 | 16384 | **0.48**       | 0.57         |
| 1 | 1024 | 32768 | **0.86**       | 1.01         |
| 1 | 1024 | 65536 | **1.61**       | 1.87         |
| 1 | 2048 | 32768 | **1.59**       | 1.83         |
| 1 | 2048 | 65536 | **3.04**       | 3.49         |
| 2 | 1024 | 32768 | **1.62**       | 1.84         |
| 4 | 1024 | 32768 | **3.12**       | 3.48         |

Run benchmarks: `cd flash-knn && pip install -e . && bash benchmarks/run_all.sh`. Results in `flash-knn/benchmarks/results_knn.jsonl`.
