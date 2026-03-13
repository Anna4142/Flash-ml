"""
KNN benchmark in KMeans style: pluggable method list, JSONL output (one line per method).
Run from flash-knn root: pip install -e . && python benchmarks/benchmark_knn_libs_v2.py ...
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from flash_knn import batch_knn_euclid, batch_knn_cosine


# ---------------------------
# Torch baselines
# ---------------------------

def torch_knn_l2(q: torch.Tensor, x: torch.Tensor, k: int):
    dists = torch.cdist(q.float(), x.float(), p=2) ** 2
    vals, idx = torch.topk(dists, k=k, dim=-1, largest=False, sorted=True)
    return vals, idx


def torch_knn_cosine(q: torch.Tensor, x: torch.Tensor, k: int):
    sims = torch.einsum("bqd,bnd->bqn", q.float(), x.float())
    vals, idx = torch.topk(sims, k=k, dim=-1, largest=True, sorted=True)
    return vals, idx


# ---------------------------
# Repo (flash-knn)
# ---------------------------

def repo_knn_l2(q: torch.Tensor, x: torch.Tensor, k: int):
    return batch_knn_euclid(q, ref=x, k=k)


def repo_knn_cosine(q: torch.Tensor, x: torch.Tensor, k: int):
    return batch_knn_cosine(q, ref=x, k=k)


# ---------------------------
# cuVS (optional)
# ---------------------------

def _cuvs_knn_l2_impl(q, x, k):
    from cuvs.neighbors import brute_force
    import cupy as cp
    B = x.shape[0]
    all_d, all_i = [], []
    for b in range(B):
        xb = cp.from_dlpack(x[b].contiguous().detach())
        qb = cp.from_dlpack(q[b].contiguous().detach())
        idx = brute_force.build(xb, metric="sqeuclidean")
        dist, neighbors = brute_force.search(idx, qb, k)
        all_d.append(torch.utils.dlpack.from_dlpack(cp.asarray(dist).toDlpack()).to(q.device))
        all_i.append(torch.utils.dlpack.from_dlpack(cp.asarray(neighbors).toDlpack()).to(q.device))
    return torch.stack(all_d, dim=0), torch.stack(all_i, dim=0)


def _cuvs_knn_cosine_impl(q, x, k):
    from cuvs.neighbors import brute_force
    import cupy as cp
    B = x.shape[0]
    all_d, all_i = [], []
    for b in range(B):
        xb = cp.from_dlpack(x[b].contiguous().detach())
        qb = cp.from_dlpack(q[b].contiguous().detach())
        idx = brute_force.build(xb, metric="cosine")
        dist, neighbors = brute_force.search(idx, qb, k)
        all_d.append(torch.utils.dlpack.from_dlpack(cp.asarray(dist).toDlpack()).to(q.device))
        all_i.append(torch.utils.dlpack.from_dlpack(cp.asarray(neighbors).toDlpack()).to(q.device))
    return torch.stack(all_d, dim=0), torch.stack(all_i, dim=0)


def make_cuvs_l2():
    def _fn(q, x, k):
        return _cuvs_knn_l2_impl(q, x, k)
    _fn.__name__ = "cuvs_l2"
    return _fn


def make_cuvs_cosine():
    def _fn(q, x, k):
        return _cuvs_knn_cosine_impl(q, x, k)
    _fn.__name__ = "cuvs_cosine"
    return _fn


# ---------------------------
# Helpers
# ---------------------------

def benchmark_one(fn, warmup=10, repeats=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def normalize_for_metric(metric, q, x):
    if metric == "cosine":
        q = F.normalize(q, dim=-1)
        x = F.normalize(x, dim=-1)
    return q, x


def benchmark_knn(B, Q, N, D, k, metric, dtype, knn_func, repeats=30):
    """Time one KNN method on random (q, x). Returns (time_ms, correct or None)."""
    q = torch.randn(B, Q, D, device="cuda", dtype=dtype)
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    q, x = normalize_for_metric(metric, q, x)
    try:
        time_ms = benchmark_one(lambda: knn_func(q, x, k), repeats=repeats)
        return time_ms, None
    except Exception as e:
        return -1.0, repr(e)


def check_correctness(metric, q, x, k, ref_vals, ref_idx, knn_func):
    """Compare knn_func output to ref (torch). Returns (idx_ok, val_ok, max_diff)."""
    vals, idx = knn_func(q, x, k)
    idx_ok = torch.equal(idx.cpu().to(ref_idx.dtype), ref_idx.cpu())
    val_ok = torch.allclose(vals.cpu().float(), ref_vals.cpu().float(), atol=1e-2, rtol=1e-2)
    max_diff = (vals.float() - ref_vals.float()).abs().max().item()
    return idx_ok, val_ok, max_diff


def benchmark_knn_all(B, Q, N, D, k, metric, dtype, knn_func_list, repeats=30, output_file="benchmarks/results_knn.jsonl"):
    """Run each method, append one JSONL line per method (KMeans style)."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    q = torch.randn(B, Q, D, device="cuda", dtype=dtype)
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    q, x = normalize_for_metric(metric, q, x)

    # Reference for correctness (torch)
    if metric == "l2":
        ref_vals, ref_idx = torch_knn_l2(q, x, k)
    else:
        ref_vals, ref_idx = torch_knn_cosine(q, x, k)

    with open(output_file, "a") as f:
        for knn_func in knn_func_list:
            name = getattr(knn_func, "__name__", str(knn_func))
            print("Benchmarking:", name)
            try:
                time_ms = benchmark_one(lambda: knn_func(q, x, k), repeats=repeats)
                correct = None
                if "repo" in name:
                    idx_ok, val_ok, max_diff = check_correctness(metric, q, x, k, ref_vals, ref_idx, knn_func)
                    correct = bool(idx_ok and val_ok)
                line = {
                    "method": name,
                    "batch_size": B,
                    "num_queries": Q,
                    "num_refs": N,
                    "dim": D,
                    "k": k,
                    "metric": metric,
                    "dtype": str(dtype).replace("torch.", ""),
                    "time_ms": round(time_ms, 4),
                    "correct": correct,
                }
                print(f"  {name}: {time_ms:.2f} ms" + (f" correct={correct}" if correct is not None else ""))
            except Exception as e:
                time_ms = -1
                line = {
                    "method": name,
                    "batch_size": B,
                    "num_queries": Q,
                    "num_refs": N,
                    "dim": D,
                    "k": k,
                    "metric": metric,
                    "dtype": str(dtype).replace("torch.", ""),
                    "time_ms": -1,
                    "error": repr(e),
                }
                print(f"  {name}: Error {e}")
            f.write(json.dumps(line) + "\n")
            f.flush()
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark KNN implementations (KMeans-style)")
    parser.add_argument("--batch-size", "-b", type=int, default=1)
    parser.add_argument("--num-queries", "-q", type=int, default=1024, help="Number of query points per batch")
    parser.add_argument("--num-refs", "-n", type=int, default=8192, help="Number of reference points per batch")
    parser.add_argument("--dim", "-d", type=int, default=128)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--metric", type=str, default="l2", choices=["l2", "cosine"])
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output-file", "-o", type=str, default="benchmarks/results_knn.jsonl")
    parser.add_argument("--sweep", action="store_true", help="Run multiple shapes (B,Q,N,D) and append all to JSONL")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    if args.metric == "l2":
        knn_func_list = [repo_knn_l2, torch_knn_l2]
        try:
            knn_func_list.append(make_cuvs_l2())
        except ImportError:
            pass
    else:
        knn_func_list = [repo_knn_cosine, torch_knn_cosine]
        try:
            knn_func_list.append(make_cuvs_cosine())
        except ImportError:
            pass

    if args.sweep:
        shapes = [
            (1, 512, 2**13, 128),
            (1, 1024, 2**14, 128),
            (1, 1024, 2**15, 128),
            (1, 1024, 2**16, 128),
            (1, 2048, 2**15, 128),
            (1, 2048, 2**16, 128),
            (2, 1024, 2**15, 128),
            (4, 1024, 2**15, 128),
        ]
        for (B, Q, N, D) in shapes:
            print(f"Shape B={B} Q={Q} N={N} D={D} k={args.k} metric={args.metric}")
            benchmark_knn_all(B, Q, N, D, args.k, args.metric, dtype, knn_func_list, repeats=args.repeats, output_file=args.output_file)
    else:
        benchmark_knn_all(
            args.batch_size, args.num_queries, args.num_refs, args.dim,
            args.k, args.metric, dtype, knn_func_list,
            repeats=args.repeats, output_file=args.output_file,
        )

    print(f"Results appended to {args.output_file}")
