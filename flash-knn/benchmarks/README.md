# Flash-KNN benchmarks

From **flash-knn** root (one level up):

```bash
pip install -e .
```

**Single config (L2):**
```bash
python benchmarks/benchmark_knn_libs.py --metric l2 --k 1 --B 1 --Q 1024 --N 8192 --D 128 --dtype fp16
```

**Single config (cosine):**
```bash
python benchmarks/benchmark_knn_libs.py --metric cosine --k 1 --B 1 --Q 1024 --N 8192 --D 128 --dtype fp16
```

**Sweep (L2, multiple shapes):**
```bash
python benchmarks/benchmark_knn_libs.py --metric l2 --k 1 --sweep --dtype fp16 --repeats 30
```

Results are written to `benchmarks/results_knn_libs.csv`. Compares flash-knn (repo) vs PyTorch and cuVS (if installed).
