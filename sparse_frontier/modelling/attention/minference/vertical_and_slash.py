# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Code adopted from https://github.com/microsoft/MInference

import math
import torch
import triton
import triton.language as tl

from sparse_frontier.modelling.attention.minference.minference import convert_vertical_slash_indexes


@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    Q, K, V, seqlens, sm_scale,
    block_count, block_offset, column_count, column_index,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    NUM_ROWS, NNZ_S, NNZ_V,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
    blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
    num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
    cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < seqlen

    for block_index in range(num_blks):
        start_n = tl.load(blks_ptr + block_index)
        cols = start_n + offs_n
        n_mask = cols < seqlen
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    for start_n in range(0, num_cols, BLOCK_N):
        n_mask = start_n + offs_n < num_cols
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & n_mask & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # write back O
    acc /= l_i[:, None]
    # acc = tl.where(m_mask, acc / l_i[:, None], 0.0)
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


def _triton_mixed_sparse_attention(
    q: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    seqlens: torch.Tensor,    # [BATCH, ]
    block_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    block_offset: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    column_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    column_index: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    sm_scale: float,
    block_size_M: int = 64,
    block_size_N: int = 64,
) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros_like(q)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    _triton_mixed_sparse_attn_fwd_kernel[grid](
        q, k, v, seqlens, sm_scale,
        block_count, block_offset, column_count, column_index,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        block_count.shape[-1], block_offset.shape[-1], column_index.shape[-1],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4, num_stages=1,
    )

    return o


def vertical_slash_sparse_attention(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    s_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    batch_size, num_heads, context_size, head_dim = query.shape
    pad = block_size_M - (context_size & (block_size_M - 1))
    query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
    key = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
    value = torch.nn.functional.pad(value, [0, 0, 0, pad, 0, 0, 0, 0])

    if head_dim not in [16, 32, 64, 128, 256, 512]:
        target_dim = 2 ** math.ceil(math.log2(head_dim)) - head_dim
        query = torch.nn.functional.pad(query, [0, target_dim, 0, 0, 0, 0, 0, 0])
        key = torch.nn.functional.pad(key, [0, target_dim, 0, 0, 0, 0, 0, 0])
        value = torch.nn.functional.pad(value, [0, target_dim, 0, 0, 0, 0, 0, 0])

    v_idx = v_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=False)[0]
    s_idx = s_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=True)[0]
    seqlens = torch.tensor([context_size], dtype=torch.int32, device=query.device)
    sm_scale = head_dim ** -0.5
    block_count, block_offset, column_count, column_index = convert_vertical_slash_indexes(
        seqlens, v_idx, s_idx, context_size, block_size_M, block_size_N,
    )
    sparsity = calc_sparsity(block_count, column_count, context_size)
    out = _triton_mixed_sparse_attention(
        query, key, value, seqlens,
        block_count, block_offset, column_count, column_index,
        sm_scale, block_size_M, block_size_N,
    )
    return out[..., :context_size, :head_dim], sparsity


def calc_sparsity(block_count, column_count, seq_len):
    block_cells = block_count.sum(dim=-1) * 64 * 64
    column_cells = column_count.sum(dim=-1) * 64
    total_cells = seq_len * (seq_len + 1) // 2
    return 1 - (block_cells + column_cells) / total_cells


def sum_over_diagonals(matrix: torch.Tensor) -> torch.Tensor:
    """Efficiently sum values along diagonals of the attention matrix.
    
    This function computes the sum of values along each diagonal of a 4D attention matrix.
    It uses an efficient strided implementation to avoid explicit diagonal extraction.
    
    Args:
        matrix: Input attention matrix of shape (batch_size, num_heads, queries, keys)
                where queries and keys are sequence lengths
    
    Returns:
        Tensor of shape (batch_size, num_heads, queries + keys - 1) containing the
        summed values for each diagonal. The diagonals are ordered from top-right
        to bottom-left, with the main diagonal at index queries-1.
    """
    batch_size, num_heads, queries, keys = matrix.shape
    zero_matrix = torch.zeros((batch_size, num_heads, queries, queries), device=matrix.device)
    matrix_padded = torch.cat((zero_matrix, matrix, zero_matrix), -1)
    
    matrix_strided = matrix_padded.as_strided(
        (batch_size, num_heads, queries, queries + keys),
        (num_heads * queries * (2 * queries + keys),
            queries * (2 * queries + keys),
            2 * queries + keys + 1, 1)
    )
    return torch.sum(matrix_strided, 2)[:, :, 1:]


def vertical_and_slash_kernel(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    vertical_size: int,
    slash_size: int, 
    last_q: int = 64, 
    inf_value: float = float('inf'), 
    topk_vertical_inf: int = 4, 
    topk_slash_inf: int = 64, 
):
    """
    Compute the vertical and slash kernel for sparse attention.

    Args:
        q: Query tensor of shape [BATCH, N_HEADS, N_CTX, D_HEAD]
        k: Key tensor of shape [BATCH, N_HEADS, N_CTX, D_HEAD]
        v: Value tensor of shape [BATCH, N_HEADS, N_CTX, D_HEAD]
        vertical_size: Size of the vertical attention
        slash_size: Size of the slash attention
        last_q: Number of last queries to consider (default: 64)
        inf_value: Value to use for infinity (default: float('inf'))
        topk_vertical_inf: Number of top-k vertical elements to set to infinity (default: 30)
        topk_slash_inf: Number of top-k slash elements to set to infinity (default: 100)

    Returns:
        Output tensor after applying vertical and slash sparse attention.
    """
    arange = torch.arange(last_q, device=q.device)
    LAST_Q_MASK = arange[None, None, :, None] >= arange[None, None, None, :]

    _, _, q_len, d = q.shape

    # Compute scaled dot-product attention
    qk = torch.einsum('bhmk, bhnk -> bhmn', q[:, :, -last_q:, :], k) / math.sqrt(d)
    qk[:, :, :, -last_q:] = torch.where(LAST_Q_MASK[..., -last_q:, -last_q:], qk[:, :, :, -last_q:], -inf_value)
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32).to(q.dtype)

    assert topk_vertical_inf <= vertical_size
    assert topk_slash_inf <= slash_size

    # Compute top verticals
    vertical = qk.sum(-2, keepdim=True)
    vertical[..., :topk_vertical_inf] = inf_value
    vertical_topk = torch.topk(vertical, vertical_size, -1).indices

    # # Compute top slashes
    slash = sum_over_diagonals(qk)[..., :-last_q + 1]
    slash[..., -topk_slash_inf:] = inf_value
    slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices

    return vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)
