import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

# Run from flash-knn root:
# pip install -e . && python benchmarks/benchmark_knn_libs_v2.py ...
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
# cuVS exact brute-force baseline
# ---------------------------

def _torch_to_cupy(t: torch.Tensor):
    import cupy as cp
    return cp.from_dlpack(t.detach())


def _cupy_to_torch(a):
    return torch.utils.dlpack.from_dlpack(a.toDlpack())


def cuvs_build_l2(x: torch.Tensor):
    from cuvs.neighbors import brute_force
    B = x.shape[0]
    indices = []
    for b in range(B):
        xb = _torch_to_cupy(x[b].contiguous())
        index = brute_force.build(xb, metric="sqeuclidean")
        indices.append(index)
    return indices


def cuvs_search_l2(indices, q: torch.Tensor, k: int):
    import cupy as cp
    from cuvs.neighbors import brute_force

    B = q.shape[0]
    all_d, all_i = [], []
    for b in range(B):
        qb = _torch_to_cupy(q[b].contiguous())
        distances, neighbors = brute_force.search(indices[b], qb, k)
        all_d.append(_cupy_to_torch(cp.asarray(distances)).to(q.device))
        all_i.append(_cupy_to_torch(cp.asarray(neighbors)).to(q.device))
    return torch.stack(all_d, dim=0), torch.stack(all_i, dim=0)


def cuvs_build_cosine(x: torch.Tensor):
    from cuvs.neighbors import brute_force
    B = x.shape[0]
    indices = []
    for b in range(B):
        xb = _torch_to_cupy(x[b].contiguous())
        index = brute_force.build(xb, metric="cosine")
        indices.append(index)
    return indices


def cuvs_search_cosine(indices, q: torch.Tensor, k: int):
    import cupy as cp
    from cuvs.neighbors import brute_force

    B = q.shape[0]
    all_d, all_i = [], []
    for b in range(B):
        qb = _torch_to_cupy(q[b].contiguous())
        distances, neighbors = brute_force.search(indices[b], qb, k)
        all_d.append(_cupy_to_torch(cp.asarray(distances)).to(q.device))
        all_i.append(_cupy_to_torch(cp.asarray(neighbors)).to(q.device))
    return torch.stack(all_d, dim=0), torch.stack(all_i, dim=0)


# ---------------------------
# Repo wrappers
# ---------------------------

def repo_knn_l2(q: torch.Tensor, x: torch.Tensor, k: int):
    return batch_knn_euclid(q, ref=x, k=k)


def repo_knn_cosine(q: torch.Tensor, x: torch.Tensor, k: int):
    return batch_knn_cosine(q, ref=x, k=k)


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


def maybe_run(fn):
    try:
        return fn(), None
    except Exception as e:
        return None, repr(e)


def normalize_for_metric(metric, q, x):
    if metric == "cosine":
        q = F.normalize(q, dim=-1)
        x = F.normalize(x, dim=-1)
    return q, x


def compare_with_torch(metric, q, x, k):
    if metric == "l2":
        ref_vals, ref_idx = torch_knn_l2(q, x, k)
        repo_vals, repo_idx = repo_knn_l2(q, x, k)
    else:
        ref_vals, ref_idx = torch_knn_cosine(q, x, k)
        repo_vals, repo_idx = repo_knn_cosine(q, x, k)

    idx_ok = torch.equal(repo_idx.cpu().to(ref_idx.dtype), ref_idx.cpu())
    val_ok = torch.allclose(repo_vals.cpu(), ref_vals.cpu(), atol=1e-2, rtol=1e-2)
    max_diff = (repo_vals.float() - ref_vals.float()).abs().max().item()
    return idx_ok, val_ok, max_diff


def full_score_matrix_bytes(B, Q, N, bytes_per_elem=4):
    # torch baselines use float32 for cdist/einsum result paths in this script
    return B * Q * N * bytes_per_elem


def fmt_gb(num_bytes):
    return num_bytes / (1024 ** 3)


def approx_dot_flops(B, Q, N, D):
    return 2.0 * B * Q * N * D


def tflops(flops, ms):
    return flops / (ms / 1000.0) / 1e12


def print_summary(row):
    print(
        f"B={row['B']} Q={row['Q']} N={row['N']} D={row['D']} "
        f"k={row['k']} metric={row['metric']} "
        f"full_matrix={row['full_matrix_gb']:.3f} GB"
    )
    print(
        f"  repo : {row['repo_ms']:.3f} ms"
        + (f" | {row['repo_tflops']:.2f} TFLOPS" if row['repo_tflops'] is not None else "")
    )
    print(
        f"  torch: {row['torch_ms']:.3f} ms"
        + (f" | {row['torch_tflops']:.2f} TFLOPS" if row['torch_tflops'] is not None else "")
    )
    if row["cuvs_build_ms"] is not None:
        print(f"  cuVS build : {row['cuvs_build_ms']:.3f} ms")
    if row["cuvs_search_ms"] is not None:
        print(f"  cuVS search: {row['cuvs_search_ms']:.3f} ms")
    print(
        f"  correctness idx={row['repo_vs_torch_idx_ok']} "
        f"vals={row['repo_vs_torch_val_ok']} "
        f"maxdiff={row['repo_vs_torch_maxdiff']:.4e}"
    )
    print(
        f"  speedups repo_vs_torch={row['speedup_vs_torch']:.3f}"
        + (
            f" repo_vs_cuvs_search={row['speedup_vs_cuvs_search']:.3f}"
            if row["speedup_vs_cuvs_search"] is not None else ""
        )
    )
    print()


def run_case(B, Q, N, D, k, metric, dtype, repeats):
    q = torch.randn(B, Q, D, device="cuda", dtype=dtype)
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    q, x = normalize_for_metric(metric, q, x)

    idx_ok, val_ok, max_diff = compare_with_torch(metric, q, x, k)

    # main timings
    if metric == "l2":
        repo_ms = benchmark_one(lambda: repo_knn_l2(q, x, k), repeats=repeats)
        torch_ms = benchmark_one(lambda: torch_knn_l2(q, x, k), repeats=repeats)

        cuvs_build_res, cuvs_build_err = maybe_run(lambda: cuvs_build_l2(x))
        if cuvs_build_res is not None:
            cuvs_build_ms = benchmark_one(lambda: cuvs_build_l2(x), repeats=max(3, repeats // 3))
            cuvs_search_res, cuvs_search_err = maybe_run(
                lambda: benchmark_one(lambda: cuvs_search_l2(cuvs_build_res, q, k), repeats=repeats)
            )
            cuvs_search_ms = cuvs_search_res
        else:
            cuvs_build_ms = None
            cuvs_search_ms = None
            cuvs_search_err = None
    else:
        repo_ms = benchmark_one(lambda: repo_knn_cosine(q, x, k), repeats=repeats)
        torch_ms = benchmark_one(lambda: torch_knn_cosine(q, x, k), repeats=repeats)

        cuvs_build_res, cuvs_build_err = maybe_run(lambda: cuvs_build_cosine(x))
        if cuvs_build_res is not None:
            cuvs_build_ms = benchmark_one(lambda: cuvs_build_cosine(x), repeats=max(3, repeats // 3))
            cuvs_search_res, cuvs_search_err = maybe_run(
                lambda: benchmark_one(lambda: cuvs_search_cosine(cuvs_build_res, q, k), repeats=repeats)
            )
            cuvs_search_ms = cuvs_search_res
        else:
            cuvs_build_ms = None
            cuvs_search_ms = None
            cuvs_search_err = None

    # useful derived metrics
    matrix_bytes = full_score_matrix_bytes(B, Q, N, bytes_per_elem=4)
    flops = approx_dot_flops(B, Q, N, D)

    row = {
        "B": B,
        "Q": Q,
        "N": N,
        "D": D,
        "k": k,
        "metric": metric,
        "dtype": str(dtype).replace("torch.", ""),
        "full_matrix_bytes": matrix_bytes,
        "full_matrix_gb": fmt_gb(matrix_bytes),
        "approx_dot_flops": flops,
        "repo_vs_torch_idx_ok": idx_ok,
        "repo_vs_torch_val_ok": val_ok,
        "repo_vs_torch_maxdiff": max_diff,
        "repo_ms": repo_ms,
        "torch_ms": torch_ms,
        "cuvs_build_ms": cuvs_build_ms,
        "cuvs_search_ms": cuvs_search_ms,
        "repo_tflops": tflops(flops, repo_ms) if repo_ms > 0 else None,
        "torch_tflops": tflops(flops, torch_ms) if torch_ms > 0 else None,
        "speedup_vs_torch": torch_ms / repo_ms if repo_ms > 0 else None,
        "speedup_vs_cuvs_search": (cuvs_search_ms / repo_ms) if (cuvs_search_ms is not None and repo_ms > 0) else None,
        "cuvs_build_error": cuvs_build_err,
        "cuvs_search_error": cuvs_search_err,
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["l2", "cosine"], default="l2")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--out", type=str, default="benchmarks/results_knn_libs_v2.csv")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--Q", type=int, default=1024)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        shapes = [
            (1, 512,   2**13, 128),
            (1, 1024,  2**14, 128),
            (1, 1024,  2**15, 128),
            (1, 1024,  2**16, 128),
            (1, 2048,  2**15, 128),
            (1, 2048,  2**16, 128),
            (2, 1024,  2**15, 128),
            (4, 1024,  2**15, 128),
        ]
    else:
        shapes = [(args.B, args.Q, args.N, args.D)]

    rows = []
    for B, Q, N, D in shapes:
        print(f"Running B={B}, Q={Q}, N={N}, D={D}, k={args.k}, metric={args.metric}")
        row = run_case(B, Q, N, D, args.k, args.metric, dtype, args.repeats)
        print_summary(row)
        rows.append(row)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
