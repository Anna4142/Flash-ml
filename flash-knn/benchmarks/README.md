# Flash-KNN benchmarks

From **flash-knn** root (one level up):

```bash
pip install -e .
```

**Single config (L2, KMeans-style):**
```bash
python benchmarks/benchmark_knn_libs_v2.py -b 1 -q 1024 -n 8192 -d 128 --k 1 --metric l2 --dtype fp16 -o benchmarks/results_knn.jsonl
```

**Single config (cosine):**
```bash
python benchmarks/benchmark_knn_libs_v2.py -b 1 -q 1024 -n 8192 -d 128 --k 1 --metric cosine -o benchmarks/results_knn.jsonl
```

**Sweep (L2, multiple shapes):**
```bash
python benchmarks/benchmark_knn_libs_v2.py --sweep --metric l2 --dtype fp16 --repeats 30 -o benchmarks/results_knn.jsonl
```

**Run all (single L2, single cosine, sweep):**
```bash
./benchmarks/run_all.sh
```

Results are appended to `benchmarks/results_knn.jsonl` (one JSONL line per method per shape). Compares repo (flash-knn), torch, and cuVS if installed; `correct` is set for repo vs torch.
