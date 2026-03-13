"""
Triton kernel: compute (BLOCK_N, BLOCK_M) block of cosine similarities for KNN.
Expects x and ref to be L2-normalized; output is dot product (max = nearest).
Used by knn_triton_impl to build similarity chunks for top-K merge.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _knn_cosine_sim_kernel(
    x_ptr,
    ref_ptr,
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
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Compute (BLOCK_N, BLOCK_M) block of cosine similarities (dot of normalized vectors)."""
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

    m_offs = tl.arange(0, BLOCK_M).to(tl.int64)
    m_mask = m_offs < M
    ref_ptrs = (
        ref_ptr + b * stride_ref_b
        + m_offs[None, :] * stride_ref_m
        + offs_d[:, None] * stride_ref_d
    )
    ref_tile = tl.load(ref_ptrs, mask=m_mask[None, :], other=0.0)

    sim = tl.dot(x_tile, ref_tile).to(tl.float32)
    sim = tl.where(m_mask[None, :], sim, -3.4e38)

    out_ptrs = (
        out_ptr + b * stride_out_b
        + n_offs[:, None] * stride_out_n
        + m_offs[None, :] * stride_out_m
    )
    tl.store(out_ptrs, sim, mask=n_mask[:, None] & m_mask[None, :])


@triton.jit
def _knn_cosine_sim_full_grid_kernel(
    x_ptr,
    ref_ptr,
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
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """One (BLOCK_N, BLOCK_M) tile of full (N, M) similarity matrix. Grid: (ceil(N/BLOCK_N), ceil(M/BLOCK_M), B)."""
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

    ref_ptrs = (
        ref_ptr + b * stride_ref_b
        + m_offs[None, :] * stride_ref_m
        + offs_d[:, None] * stride_ref_d
    )
    ref_tile = tl.load(ref_ptrs, mask=m_mask[None, :], other=0.0)

    sim = tl.dot(x_tile, ref_tile).to(tl.float32)
    sim = tl.where(m_mask[None, :], sim, -3.4e38)

    out_ptrs = (
        out_ptr + b * stride_out_b
        + n_offs[:, None] * stride_out_n
        + m_offs[None, :] * stride_out_m
    )
    tl.store(out_ptrs, sim, mask=n_mask[:, None] & m_mask[None, :])


def knn_cosine_sim_full_matrix_triton(
    x: torch.Tensor,
    ref: torch.Tensor,
    out: torch.Tensor,
    BLOCK_N: int = 256,
    BLOCK_M: int = 512,
) -> None:
    """Fill full (B, N, M) similarity matrix in one 2D grid launch. out must be (B, N, M)."""
    B, N, D = x.shape
    assert ref.shape == (B, out.shape[2], D)
    M = ref.shape[1]
    assert out.shape == (B, N, M)
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M), B)
    _knn_cosine_sim_full_grid_kernel[grid](
        x,
        ref,
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
        stride_out_b=out.stride(0),
        stride_out_n=out.stride(1),
        stride_out_m=out.stride(2),
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
    )


def knn_cosine_sim_chunk_triton(
    x: torch.Tensor,
    ref_chunk: torch.Tensor,
    out: torch.Tensor,
    BLOCK_N: int = 128,
    BLOCK_M: int = 128,
) -> None:
    """
    Fill out (B, N, chunk_M) with cosine similarities for this ref chunk.
    x and ref_chunk should be L2-normalized along last dim.
    """
    B, N, D = x.shape
    chunk_M = ref_chunk.shape[1]
    assert ref_chunk.shape == (B, chunk_M, D)
    assert out.shape == (B, N, chunk_M)
    grid = (triton.cdiv(N, BLOCK_N), B)
    _knn_cosine_sim_kernel[grid](
        x,
        ref_chunk,
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
        stride_out_b=out.stride(0),
        stride_out_n=out.stride(1),
        stride_out_m=out.stride(2),
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
    )
