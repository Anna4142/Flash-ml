"""
Triton kernel: compute (BLOCK_N, BLOCK_M) block of squared Euclidean distances
for KNN. Same pattern as flash-kmeans assign: x_sq + ref_sq - 2*x@ref.
Used by knn_triton_impl to build distance chunks for top-K merge.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _knn_euclid_dist_kernel(
    x_ptr,
    ref_ptr,
    x_sq_ptr,
    ref_sq_ptr,
    out_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_ref_b: tl.constexpr,
    stride_ref_m: tl.constexpr,
    stride_ref_d: tl.constexpr,
    stride_xsq_b: tl.constexpr,
    stride_xsq_n: tl.constexpr,
    stride_refsq_b: tl.constexpr,
    stride_refsq_m: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Compute a (BLOCK_N, BLOCK_M) block of squared Euclidean distances.
    Grid: (ceil(N/BLOCK_N), B). ref_ptr is the current ref chunk [m_start, m_start+BLOCK_M).
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    b = pid_b.to(tl.int64)

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_offs = n_offs.to(tl.int64)
    n_mask = n_offs < N

    offs_d = tl.arange(0, D).to(tl.int64)

    x_ptrs = (
        x_ptr + b * stride_x_b
        + n_offs[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d
    )
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
    x_sq_ptrs = x_sq_ptr + b * stride_xsq_b + n_offs * stride_xsq_n
    x_sq_tile = tl.load(x_sq_ptrs, mask=n_mask, other=0.0).to(tl.float32)

    m_offs = tl.arange(0, BLOCK_M).to(tl.int64)
    m_mask = m_offs < M

    ref_ptrs = (
        ref_ptr + b * stride_ref_b
        + m_offs[None, :] * stride_ref_m
        + offs_d[:, None] * stride_ref_d
    )
    ref_tile = tl.load(ref_ptrs, mask=m_mask[None, :], other=0.0)
    ref_sq_ptrs = ref_sq_ptr + b * stride_refsq_b + m_offs * stride_refsq_m
    ref_sq_tile = tl.load(ref_sq_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    cross = tl.dot(x_tile, ref_tile).to(tl.float32)
    dist = x_sq_tile[:, None] + ref_sq_tile[None, :] - 2.0 * cross
    dist = tl.maximum(dist, 0.0)
    dist = tl.where(m_mask[None, :], dist, 3.4e38)

    out_ptrs = (
        out_ptr + b * stride_out_b
        + n_offs[:, None] * stride_out_n
        + m_offs[None, :] * stride_out_m
    )
    tl.store(out_ptrs, dist, mask=n_mask[:, None] & m_mask[None, :])


@triton.jit
def _knn_euclid_dist_full_grid_kernel(
    x_ptr,
    ref_ptr,
    x_sq_ptr,
    ref_sq_ptr,
    out_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_ref_b: tl.constexpr,
    stride_ref_m: tl.constexpr,
    stride_ref_d: tl.constexpr,
    stride_xsq_b: tl.constexpr,
    stride_xsq_n: tl.constexpr,
    stride_refsq_b: tl.constexpr,
    stride_refsq_m: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Compute one (BLOCK_N, BLOCK_M) tile of the full (N, M) distance matrix.
    Grid: (ceil(N/BLOCK_N), ceil(M/BLOCK_M), B). One launch fills entire matrix.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)
    b = pid_b.to(tl.int64)

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_offs = n_offs.to(tl.int64)
    n_mask = n_offs < N

    m_start = pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_offs = m_offs.to(tl.int64)
    m_mask = m_offs < M

    offs_d = tl.arange(0, D).to(tl.int64)

    x_ptrs = (
        x_ptr + b * stride_x_b
        + n_offs[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d
    )
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
    x_sq_ptrs = x_sq_ptr + b * stride_xsq_b + n_offs * stride_xsq_n
    x_sq_tile = tl.load(x_sq_ptrs, mask=n_mask, other=0.0).to(tl.float32)

    ref_ptrs = (
        ref_ptr + b * stride_ref_b
        + m_offs[None, :] * stride_ref_m
        + offs_d[:, None] * stride_ref_d
    )
    ref_tile = tl.load(ref_ptrs, mask=m_mask[None, :], other=0.0)
    ref_sq_ptrs = ref_sq_ptr + b * stride_refsq_b + m_offs * stride_refsq_m
    ref_sq_tile = tl.load(ref_sq_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    cross = tl.dot(x_tile, ref_tile).to(tl.float32)
    dist = x_sq_tile[:, None] + ref_sq_tile[None, :] - 2.0 * cross
    dist = tl.maximum(dist, 0.0)
    dist = tl.where(m_mask[None, :], dist, 3.4e38)

    out_ptrs = (
        out_ptr + b * stride_out_b
        + n_offs[:, None] * stride_out_n
        + m_offs[None, :] * stride_out_m
    )
    tl.store(out_ptrs, dist, mask=n_mask[:, None] & m_mask[None, :])


def knn_euclid_dist_full_matrix_triton(
    x: torch.Tensor,
    ref: torch.Tensor,
    x_sq: torch.Tensor,
    ref_sq: torch.Tensor,
    out: torch.Tensor,
    BLOCK_N: int = 256,
    BLOCK_M: int = 512,
) -> None:
    """Fill full (B, N, M) distance matrix in one 2D grid launch. out must be (B, N, M)."""
    B, N, D = x.shape
    assert ref.shape == (B, out.shape[2], D)
    M = ref.shape[1]
    assert out.shape == (B, N, M)
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M), B)
    _knn_euclid_dist_full_grid_kernel[grid](
        x,
        ref,
        x_sq,
        ref_sq,
        out,
        N=N,
        M=M,
        D=D,
        stride_x_b=x.stride(0),
        stride_x_n=x.stride(1),
        stride_x_d=x.stride(2),
        stride_ref_b=ref.stride(0),
        stride_ref_m=ref.stride(1),
        stride_ref_d=ref.stride(2),
        stride_xsq_b=x_sq.stride(0),
        stride_xsq_n=x_sq.stride(1),
        stride_refsq_b=ref_sq.stride(0),
        stride_refsq_m=ref_sq.stride(1),
        stride_out_b=out.stride(0),
        stride_out_n=out.stride(1),
        stride_out_m=out.stride(2),
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
    )


def knn_euclid_dist_chunk_triton(
    x: torch.Tensor,
    ref_chunk: torch.Tensor,
    x_sq: torch.Tensor,
    ref_sq_chunk: torch.Tensor,
    out: torch.Tensor,
    BLOCK_N: int = 128,
    BLOCK_M: int = 128,
) -> None:
    """
    Fill out (B, N, chunk_M) with squared L2 distances for this ref chunk.
    Caller passes pre-sliced ref_chunk and ref_sq_chunk; out must be (B, N, ref_chunk.shape[1]).
    """
    B, N, D = x.shape
    chunk_M = ref_chunk.shape[1]
    assert ref_chunk.shape == (B, chunk_M, D)
    assert out.shape == (B, N, chunk_M)
    grid = (triton.cdiv(N, BLOCK_N), B)
    _knn_euclid_dist_kernel[grid](
        x,
        ref_chunk,
        x_sq,
        ref_sq_chunk,
        out,
        B=B,
        N=N,
        M=chunk_M,
        D=D,
        stride_x_b=x.stride(0),
        stride_x_n=x.stride(1),
        stride_x_d=x.stride(2),
        stride_ref_b=ref_chunk.stride(0),
        stride_ref_m=ref_chunk.stride(1),
        stride_ref_d=ref_chunk.stride(2),
        stride_xsq_b=x_sq.stride(0),
        stride_xsq_n=x_sq.stride(1),
        stride_refsq_b=ref_sq_chunk.stride(0),
        stride_refsq_m=ref_sq_chunk.stride(1),
        stride_out_b=out.stride(0),
        stride_out_n=out.stride(1),
        stride_out_m=out.stride(2),
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
    )
