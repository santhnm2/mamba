# Copyright (c) 2024, Tri Dao, Albert Gu.

"""We want triton==2.1.0 or 2.2.0 for this
"""

import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['dim'],
)
@triton.jit
def _state_passing_fwd_kernel(
    # Pointers to matrices
    states_ptr, out_ptr, final_states_ptr, dA_cs_ptr, initstates_ptr, seq_idx_ptr,
    # Matrix dimensions
    dim, nchunks, seqlen, chunk_size,
    # Strides
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_dim,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_out_dim,
    stride_final_states_batch, stride_final_states_head, stride_final_states_dim,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head,
    stride_initstates_batch, stride_initstates_head, stride_initstates_dim,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # Meta-parameters
    HAS_INITSTATES: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    dA_cs_ptr += pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    request_id_init = pid_b
    request_id_final = pid_b
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch
        request_id_init = tl.load(seq_idx_ptr)
        request_id_final = tl.load(seq_idx_ptr + (seqlen - 1) * stride_seq_idx_seqlen)

    final_states_ptr += request_id_final * stride_final_states_batch + pid_h * stride_final_states_head
    if HAS_INITSTATES:
        initstates_ptr += request_id_init * stride_initstates_batch + pid_h * stride_initstates_head

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    final_states_ptrs = final_states_ptr + offs_m * stride_final_states_dim

    if not HAS_INITSTATES:
        states = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    else:
        initstates_ptrs = initstates_ptr + offs_m * stride_initstates_dim
        states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    tl.store(out_ptrs, states, mask=offs_m < dim)
    out_ptrs += stride_out_chunk
    seq_idx = 0
    if HAS_SEQ_IDX:
        # We need to make sure that the first chunk is not reset, so we load the seq_idx of the first element
        seq_idx = tl.load(seq_idx_ptr)

    for c in range(nchunks):
        new_states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            seq_idx_new = tl.load(seq_idx_ptr + (min((c + 1) * chunk_size, seqlen) - 1) * stride_seq_idx_seqlen)
            scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
            seq_idx = seq_idx_new
        states = scale * states + new_states
        if c < nchunks - 1:
            tl.store(out_ptrs, states, mask=offs_m < dim)
        else:
            tl.store(final_states_ptrs, states, mask=offs_m < dim)
        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['dim'],
)
@triton.jit
def _state_passing_bwd_kernel(
    # Pointers to matrices
    dout_ptr, out_ptr, dA_cs_ptr, dfinal_states_ptr, seq_idx_ptr,
    dstates_ptr, ddA_cs_ptr, dinitstates_ptr, states_converted_ptr,
    # Matrix dimensions
    dim, nchunks, seqlen, chunk_size,
    # Strides
    stride_dout_batch, stride_dout_chunk, stride_dout_head, stride_dout_dim,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_out_dim,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head,
    stride_dfinal_states_batch, stride_dfinal_states_head, stride_dfinal_states_dim,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_dstates_batch, stride_dstates_chunk, stride_dstates_head, stride_dstates_dim,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head,
    stride_dinitstates_batch, stride_dinitstates_head, stride_dinitstates_dim,
    # Meta-parameters
    CONVERT_STATES: tl.constexpr,
    HAS_DFINAL_STATES: tl.constexpr,
    HAS_DINITSTATES: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    dstates_ptr += pid_b * stride_dstates_batch + pid_h * stride_dstates_head + (nchunks - 1) * stride_dstates_chunk
    dA_cs_ptr += pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head + (nchunks - 1) * stride_dA_cs_chunk
    ddA_cs_ptr += pid_b * stride_ddA_cs_batch + pid_h * stride_ddA_cs_head + (nchunks - 1) * stride_ddA_cs_chunk + pid_m
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head + (nchunks - 1) * stride_out_chunk
    dout_ptr += pid_b * stride_dout_batch + pid_h * stride_dout_head + (nchunks - 1) * stride_dout_chunk
    if CONVERT_STATES:
        states_converted_ptr += pid_b * stride_out_batch + pid_h * stride_out_head + (nchunks - 1) * stride_out_chunk

    request_id_init = pid_b
    request_id_final = pid_b
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch
        request_id_init = tl.load(seq_idx_ptr)
        request_id_final = tl.load(seq_idx_ptr + (seqlen - 1) * stride_seq_idx_seqlen)

    if HAS_DFINAL_STATES:
        dfinal_states_ptr += request_id_final * stride_dfinal_states_batch + pid_h * stride_dfinal_states_head
    if HAS_DINITSTATES:
        dinitstates_ptr += request_id_init * stride_dinitstates_batch + pid_h * stride_dinitstates_head

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dstates_ptrs = dstates_ptr + offs_m * stride_dstates_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    dout_ptrs = dout_ptr + offs_m * stride_dout_dim
    if CONVERT_STATES:
        states_converted_ptrs = states_converted_ptr + offs_m * stride_out_dim

    if HAS_DFINAL_STATES:
        dstates = tl.load(dfinal_states_ptr + offs_m * stride_dfinal_states_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
    else:
        dstates = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    tl.store(dstates_ptrs, dstates, mask=offs_m < dim)
    if HAS_SEQ_IDX:
        seq_idx = tl.load(seq_idx_ptr + (seqlen - 1) * stride_seq_idx_seqlen)
    dstates_ptrs -= stride_dstates_chunk
    for c in range(nchunks - 1):
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            seq_idx_new = tl.load(seq_idx_ptr + (((nchunks - c - 1) * chunk_size - 1) * stride_seq_idx_seqlen))
            scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
            seq_idx = seq_idx_new
        out = tl.load(out_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if CONVERT_STATES:
            tl.store(states_converted_ptrs, out, mask=offs_m < dim)
        ddA = tl.sum(out * dstates) * scale
        tl.store(ddA_cs_ptr, ddA)
        dout = tl.load(dout_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dstates = scale * dstates + dout
        tl.store(dstates_ptrs, dstates, mask=offs_m < dim)
        dout_ptrs -= stride_dout_chunk
        dstates_ptrs -= stride_dstates_chunk
        dA_cs_ptr -= stride_dA_cs_chunk
        ddA_cs_ptr -= stride_ddA_cs_chunk
        out_ptrs -= stride_out_chunk
        if CONVERT_STATES:
            states_converted_ptrs -= stride_out_chunk
    if CONVERT_STATES:
        out = tl.load(out_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        tl.store(states_converted_ptrs, out, mask=offs_m < dim)
    if not HAS_DINITSTATES:
        tl.store(ddA_cs_ptr, 0.0)
    else:
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            # We check that seq_idx of the element at chunk_size - 1 is the same as the seq_idx of the
            # element at 0. This is only true if the sequence has length > chunk_size - 1.
            # Otherwise this logic is faulty, but it's ok since we only use this logic
            # for the backward pass of the state passing, and the states are only passed for
            # sequence length > chunk_size.
            seq_idx_new = tl.load(seq_idx_ptr + ((chunk_size - 1) * stride_seq_idx_seqlen))
            scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
        out = tl.load(out_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        ddA = tl.sum(out * dstates) * scale
        tl.store(ddA_cs_ptr, ddA)
        dout = tl.load(dout_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dstates = scale * dstates + dout
        tl.store(dinitstates_ptr + offs_m * stride_dinitstates_dim, dstates, mask=offs_m < dim)


def _state_passing_fwd(states, dA_chunk_cumsum, initial_states=None, seq_idx=None, chunk_size=None,
                       out_dtype=None):
    batch_states, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch_states, nheads, nchunks)

    if seq_idx is None:
        # Standard case: batch dimension is derived from the `states` tensor.
        batch = batch_states
        if initial_states is not None:
            assert initial_states.shape == (batch, nheads, dim)
    else:
        # Variable-length case: `states` is a single long sequence (batch_states=1).
        assert batch_states == 1, "batch_states must be 1 when seq_idx is provided"
        assert chunk_size is not None, "chunk_size must be provided when seq_idx is provided"

        seqlen = seq_idx.shape[-1]

        if initial_states is not None:
            batch = initial_states.shape[0]
            assert initial_states.shape == (batch, nheads, dim)
            # Sanity check that sequence indices are within the batch size.
            assert seq_idx.max() < batch, "seq_idx contains an index that is out of bounds for the provided initial_states"
        else:
            # If initial_states is not provided, infer the batch size from seq_idx.
            # The kernel will handle the zero-initialization of states.
            batch = int(seq_idx.max().item() + 1)

    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((batch_states, nchunks, nheads, dim), device=states.device, dtype=out_dtype)
    final_states = torch.empty((batch, nheads, dim), device=states.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), batch_states, nheads)
    with torch.cuda.device(states.device.index):
        _state_passing_fwd_kernel[grid](
            states, out, final_states, dA_chunk_cumsum, initial_states, seq_idx,
            dim, nchunks, seqlen if seq_idx is not None else 0, chunk_size if seq_idx is not None else 0,
            states.stride(0), states.stride(1), states.stride(2), states.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            final_states.stride(0), final_states.stride(1), final_states.stride(2),
            dA_chunk_cumsum.stride(0), dA_chunk_cumsum.stride(2), dA_chunk_cumsum.stride(1),
            *((initial_states.stride(0), initial_states.stride(1), initial_states.stride(2))
              if initial_states is not None else (0, 0, 0)),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_INITSTATES=initial_states is not None,
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return out, final_states


def _state_passing_bwd(
        states, dA_chunk_cumsum, dout, dfinal_states=None, seq_idx=None, has_initial_states=None,
        dstates_dtype=None, states_dtype=None, chunk_size=None
):
    """
    states contains the initial_states at index 0. The final states are not included in states.
    """
    batch_dout, nchunks, nheads, dim = dout.shape
    batch = batch_dout if seq_idx is None else dfinal_states.shape[0]
    if seq_idx is not None:
        assert dfinal_states is not None
        assert batch_dout == 1, "batch_dout must be 1 when seq_idx is provided"
    assert dA_chunk_cumsum.shape == (batch_dout, nheads, nchunks)
    assert states.shape == (batch_dout, nchunks, nheads, dim)
    if seq_idx is not None:
        assert chunk_size is not None
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch_dout, seqlen)
    dstates = torch.empty_like(dout, dtype=dstates_dtype if dstates_dtype is not None else dout.dtype)
    if states_dtype is not None and states_dtype != states.dtype:
        states_converted = torch.empty_like(states, dtype=dstates_dtype if dstates_dtype is not None else dout.dtype)
        assert states_converted.stride() == states.stride()
    else:
        states_converted = None
    if has_initial_states:
        dinitstates = torch.empty((batch, nheads, dim), device=states.device, dtype=dstates.dtype)
    else:
        dinitstates = None
    if dfinal_states is not None:
        assert dfinal_states.shape == (batch, nheads, dim)
    BLOCK_SIZE_min = 64
    n_blocks = (dim + BLOCK_SIZE_min - 1) // BLOCK_SIZE_min
    ddA_chunk_cumsum = torch.empty(batch_dout, nheads, nchunks, n_blocks,
                                     dtype=torch.float32, device=dA_chunk_cumsum.device)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), batch_dout, nheads)
    with torch.cuda.device(dout.device.index):
        _state_passing_bwd_kernel[grid](
            dout, states, dA_chunk_cumsum, dfinal_states, seq_idx,
            dstates, ddA_chunk_cumsum, dinitstates, states_converted,
            dim, nchunks, seqlen if seq_idx is not None else 0, chunk_size if seq_idx is not None else 0,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3),
            dA_chunk_cumsum.stride(0), dA_chunk_cumsum.stride(2), dA_chunk_cumsum.stride(1),
            *((dfinal_states.stride(0), dfinal_states.stride(1), dfinal_states.stride(2))
               if dfinal_states is not None else (0, 0, 0)),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3),
            ddA_chunk_cumsum.stride(0), ddA_chunk_cumsum.stride(2), ddA_chunk_cumsum.stride(1),
            *((dinitstates.stride(0), dinitstates.stride(1), dinitstates.stride(2))
              if dinitstates is not None else (0, 0, 0)),
            CONVERT_STATES=states_converted is not None,
            HAS_DFINAL_STATES=dfinal_states is not None,
            HAS_DINITSTATES=dinitstates is not None,
            HAS_SEQ_IDX=seq_idx is not None,
        )
    BLOCK_SIZE_actual = _state_passing_bwd_kernel.best_config.kwargs["BLOCK_SIZE"]
    n_valid_blocks = (dim + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
    ddA_chunk_cumsum = ddA_chunk_cumsum[..., :n_valid_blocks].sum(dim=-1).to(dtype=dA_chunk_cumsum.dtype)
    if states_dtype is not None and states_dtype == states.dtype:
        states_converted = states
    return (dstates, ddA_chunk_cumsum, dinitstates) if states_dtype is None else (dstates, ddA_chunk_cumsum, dinitstates, states_converted)


class StatePassingFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, states, dA_chunk_cumsum, initial_states=None, seq_idx=None, chunk_size=None):
        batch, nchunks, nheads, dim = states.shape
        if states.stride(-1) != 1:
            states = states.contiguous()
        # TODO: dA_chunk_cumsum could be contiguous
        out, final_states = _state_passing_fwd(states, dA_chunk_cumsum, initial_states, seq_idx,
                                               chunk_size)
        ctx.save_for_backward(out, dA_chunk_cumsum, seq_idx)
        ctx.has_initial_states = initial_states is not None
        ctx.chunk_size = chunk_size
        return out, final_states

    @staticmethod
    def backward(ctx, dout, dfinal_states):
        out, dA_chunk_cumsum, seq_idx = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        dstates, ddA_chunk_cumsum, dinitstates = _state_passing_bwd(
            out, dA_chunk_cumsum, dout, dfinal_states=dfinal_states, seq_idx=seq_idx,
            has_initial_states=ctx.has_initial_states, chunk_size=ctx.chunk_size
        )
        return dstates, ddA_chunk_cumsum, dinitstates, None, None


def state_passing(states, dA_chunk_cumsum, initial_states=None, seq_idx=None, chunk_size=None):
    """
    Argument:
        states: (batch, nchunks, nheads, dim)
        dA_chunk_cumsum: (batch, nheads, nchunks)
        initial_states: (batch, nheads, dim)
    Return:
        out: (batch, nchunks, nheads, dim)
        final_states: (batch, nheads, dim)
    """
    return StatePassingFn.apply(states, dA_chunk_cumsum, initial_states, seq_idx, chunk_size)


def state_passing_ref(states, dA_chunk_cumsum, initial_states=None, seq_idx=None, chunk_size=None):
    """
    Argument:
        states: (batch, nchunks, nheads, dim)
        dA_chunk_cumsum: (batch, nheads, nchunks)
        initial_states: (batch, nheads, dim)
    Return:
        out: (batch, nchunks, nheads, dim)
        final_states: (batch, nheads, dim)
    """
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, 0])
    if seq_idx is None:
        states = torch.cat([rearrange(initial_states, "b h d -> b 1 h d"), states], dim=1)
        dA_chunk_cumsum = F.pad(dA_chunk_cumsum, (1, 0))
        dA_chunk_cumsum = torch.cumsum(dA_chunk_cumsum, dim=-1)
        nchunks = dA_chunk_cumsum.shape[-1]
        # (batch, nheads, nchunks, nchunks)
        dt_chunk_segment_sum = dA_chunk_cumsum[:, :, :, None] - dA_chunk_cumsum[:, :, None, :]
        # (batch, nheads, nchunks, nchunks)
        decay_chunk = torch.exp(dt_chunk_segment_sum)
        causal_mask = torch.tril(torch.ones(nchunks, nchunks, device=states.device, dtype=bool), diagonal=0)
        decay_chunk = decay_chunk.masked_fill(~causal_mask, 0)
        out = torch.einsum("bhzc,bchd->bzhd", decay_chunk.to(dtype=states.dtype), states)
        return out[:, :-1], out[:, -1]
    else:
        assert states.shape[0] == 1
        states = torch.cat([rearrange(initial_states, "b h d -> b 1 h d"), states[0]], dim=0)
        # states: (batch_size + 1, nheads, dim)
        batch_size = initial_states.shape[0]
        out_chunks = []
        h = initial_states.clone()
        seq_idx_chunks = seq_idx[0].split(chunk_size)
        for i, (states_chunk, dA_cs_chunk, seq_idx_chunk) in enumerate(zip(states.split(1)[1:], dA_chunk_cumsum[0].t().split(1), seq_idx_chunks)):
            # h: (batch_size, nheads, dim), states_chunk: (1, nheads, dim), dA_cs_chunk: (nheads, 1)
            # seq_idx_chunk: (chunk_size,)
            if i > 0:
                is_same_seq = (seq_idx_chunk[0] == seq_idx_chunks[i - 1][-1]).unsqueeze(-1).unsqueeze(-1)
                h = torch.where(is_same_seq, h, 0.0)
            h = h * torch.exp(dA_cs_chunk).unsqueeze(0) + states_chunk
            out_chunks.append(h)
        out = torch.stack(out_chunks, dim=1)
        final_states = h
        return out, final_states
