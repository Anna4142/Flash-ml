#!/bin/bash
# Run from flash-knn root: ./benchmarks/run_all.sh
set -e
python benchmarks/benchmark_knn_libs.py --metric l2 --k 1 --B 1 --Q 1024 --N 8192 --D 128 --dtype fp16
python benchmarks/benchmark_knn_libs.py --metric cosine --k 1 --B 1 --Q 1024 --N 8192 --D 128 --dtype fp16
python benchmarks/benchmark_knn_libs.py --metric l2 --k 1 --sweep --dtype fp16 --repeats 30
echo "Done. Results in benchmarks/results_knn_libs.csv"
