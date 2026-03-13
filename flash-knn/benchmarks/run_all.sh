#!/bin/bash
# Run from flash-knn root: ./benchmarks/run_all.sh (KMeans-style: method list + JSONL)
set -e
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
  PYTHON=python
fi
OUT="benchmarks/results_knn.jsonl"
rm -f "$OUT"
"$PYTHON" benchmarks/benchmark_knn_libs_v2.py -b 1 -q 1024 -n 8192 -d 128 --k 1 --metric l2 --dtype fp16 -o "$OUT"
"$PYTHON" benchmarks/benchmark_knn_libs_v2.py -b 1 -q 1024 -n 8192 -d 128 --k 1 --metric cosine --dtype fp16 -o "$OUT"
"$PYTHON" benchmarks/benchmark_knn_libs_v2.py --sweep --metric l2 --dtype fp16 --repeats 30 -o "$OUT"
echo "Done. Results in $OUT"
