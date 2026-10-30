# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ruff: noqa: E741
"""Explicit block-sparse attention over tile-padded sequences (Pallas TPU).

The sequence is cut into tiles of ``block_size`` rows. A boolean
``block_map[b, h, i, j]`` says whether query tile ``i`` attends to key tile
``j``, and ``block_sizes[j]`` how many leading rows of key tile ``j`` are
valid. This is the contract of video sparse attention (VSA) style kernels,
where the selection is data dependent (a per-row top-k of pooled scores).

Layout of the computation, per (batch * head, query tile) grid program:

- The whole K and V of the current head are copied into VMEM once, at the
  first query tile of that head, and key tiles are read from there with
  dynamic slices. This avoids a grid step (and its DMA) per selected tile.
- The selected key tiles of the query tile come from an index table (sorted
  tile ids, one row per program, staged in SMEM), ``kb`` tiles per loop
  iteration, so each iteration is one ``[block_size, kb * block_size]``
  score block on the MXU.
- Keys past a tile's valid length, and table slots past the row's count,
  are masked. The softmax is online, in f32; QK^T and PV are bf16 MXU
  products with f32 accumulation.
- A query tile that selects no valid key returns zeros.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_MASK = -1e30
# K and V of one head live in VMEM; leave room for the q / out blocks, the
# score block and compiler scratch.
_VMEM_HEADROOM_BYTES = 16 * 1024 * 1024


def kv_vmem_bytes(seq_len: int, head_dim: int, dtype) -> int:
    """VMEM taken by one head's K and V."""
    return 2 * seq_len * head_dim * jnp.dtype(dtype).itemsize


def is_supported(
    seq_len: int, head_dim: int, dtype, block_size: int, vmem_capacity_bytes: int
) -> bool:
    """Whether one head's K and V fit in VMEM next to the working set."""
    return (
        block_size % 128 == 0
        and head_dim % 128 == 0
        and seq_len % block_size == 0
        and kv_vmem_bytes(seq_len, head_dim, dtype) + _VMEM_HEADROOM_BYTES
        <= vmem_capacity_bytes
    )


def block_tables(block_map: jax.Array, kb: int) -> tuple[jax.Array, jax.Array]:
    """Selected key tiles per (batch * head, query tile) program.

    Returns ``(table [P, 1, kmax] int32, count [P] int32)`` with
    ``P = B * H * blocks``: the selected key tile ids in ascending order,
    padded with tile 0 up to ``kmax`` (a multiple of ``kb``); the kernel masks
    slots at or past ``count``.
    """
    batch, heads, blocks, key_blocks = block_map.shape
    sel = block_map.reshape(batch * heads * blocks, key_blocks)
    count = jnp.sum(sel, axis=-1, dtype=jnp.int32)
    # Stable: selected tiles first, in ascending tile order.
    order = jnp.argsort(jnp.logical_not(sel), axis=-1, stable=True).astype(jnp.int32)
    kmax = kb * math.ceil(key_blocks / kb)
    if kmax > key_blocks:
        order = jnp.pad(order, ((0, 0), (0, kmax - key_blocks)))
    table = jnp.where(jnp.arange(kmax)[None, :] < count[:, None], order, 0)
    return table.reshape(-1, 1, kmax), count


@functools.partial(
    jax.jit,
    static_argnames=("scale", "block_size", "kb", "vmem_limit_bytes", "interpret"),
)
def block_sparse_attention_bshd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    block_map: jax.Array,
    block_sizes: jax.Array,
    *,
    scale: float,
    block_size: int = 128,
    kb: int = 8,
    vmem_limit_bytes: int | None = None,
    interpret=False,
) -> jax.Array:
    """Block-sparse attention on (B, S, H, D) inputs with S = blocks * block_size.

    Args:
        q, k, v: bf16 (B, S, H, D), padded to whole tiles.
        block_map: bool (B, H, blocks, blocks); query tile i attends to key
            tile j where True.
        block_sizes: int32 (blocks,); valid leading rows of each key tile.
        scale: softmax scale.
        block_size: rows per tile (a multiple of 128).
        kb: key tiles per inner-loop iteration. 8 was the fastest of 2 / 4 / 8
            on v6e at 7 heads x 800 tiles with ~10% selected (40.1 / 53.6 /
            100.7 ms per call for 8 / 4 / 2).
        vmem_limit_bytes: Mosaic VMEM limit; defaults to the K/V scratch plus
            headroom.
        interpret: forwarded to ``pl.pallas_call`` (tests run on CPU with it).

    Returns:
        (B, S, H, D) output in q's dtype. Rows of a query tile that selects no
        valid key are zero.
    """
    batch, seq_len, heads, dim = q.shape
    if seq_len % block_size:
        raise ValueError(
            f"sequence length {seq_len} is not a multiple of block_size {block_size}"
        )
    blocks = seq_len // block_size
    if block_map.shape != (batch, heads, blocks, blocks):
        raise ValueError(
            f"block_map shape {block_map.shape} != {(batch, heads, blocks, blocks)}"
        )
    if block_sizes.shape != (blocks,):
        raise ValueError(f"block_sizes shape {block_sizes.shape} != {(blocks,)}")
    bh = batch * heads

    table, count = block_tables(block_map, kb)
    kmax = table.shape[-1]

    def to_heads(x):
        return jnp.swapaxes(x, 1, 2).reshape(bh, seq_len, dim)

    qh, kh, vh = to_heads(q), to_heads(k), to_heads(v)
    width = kb * block_size

    def kernel(
        count_ref, sizes_ref, table_ref, q_ref, k_hbm, v_hbm, o_ref, k_vmem, v_vmem, sem
    ):
        h = pl.program_id(0)
        i = pl.program_id(1)

        @pl.when(i == 0)
        def _load_head():
            ck = pltpu.make_async_copy(k_hbm.at[h], k_vmem, sem.at[0])
            cv = pltpu.make_async_copy(v_hbm.at[h], v_vmem, sem.at[1])
            ck.start()
            cv.start()
            ck.wait()
            cv.wait()

        cnt = count_ref[h * blocks + i]
        qb = q_ref[0]  # (block_size, D)
        seg = jax.lax.broadcasted_iota(jnp.int32, (1, width), 1) // block_size
        col_in = jax.lax.broadcasted_iota(jnp.int32, (1, width), 1) % block_size

        def body(g, carry):
            m, l, acc = carry
            ks, vs = [], []
            lim = jnp.zeros((1, width), jnp.int32)
            for t in range(kb):
                slot = g * kb + t
                tile = table_ref[0, 0, slot]
                start = pl.multiple_of(tile * block_size, block_size)
                ks.append(k_vmem[pl.ds(start, block_size), :])
                vs.append(v_vmem[pl.ds(start, block_size), :])
                valid = jnp.where(slot < cnt, sizes_ref[tile], 0)
                lim = jnp.where(seg == t, valid, lim)
            kt = jnp.concatenate(ks, axis=0) if kb > 1 else ks[0]
            vt = jnp.concatenate(vs, axis=0) if kb > 1 else vs[0]
            s = jax.lax.dot_general(
                qb, kt, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            )
            s = jnp.where(col_in < lim, s * scale, _MASK)
            m_new = jnp.maximum(m, s.max(axis=1, keepdims=True))
            p = jnp.exp(s - m_new)
            alpha = jnp.exp(m - m_new)
            l_new = alpha * l + p.sum(axis=1, keepdims=True)
            acc_new = alpha * acc + jnp.dot(
                p.astype(vt.dtype), vt, preferred_element_type=jnp.float32
            )
            return m_new, l_new, acc_new

        init = (
            jnp.full((block_size, 1), _MASK, jnp.float32),
            jnp.zeros((block_size, 1), jnp.float32),
            jnp.zeros((block_size, dim), jnp.float32),
        )
        m, l, acc = jax.lax.fori_loop(0, (cnt + kb - 1) // kb, body, init)
        # A row whose groups were all masked keeps m at the mask value: its
        # l and acc only hold the masked placeholder terms, so return zeros.
        has_key = m > _MASK / 2
        o_ref[0] = jnp.where(has_key, acc / jnp.where(has_key, l, 1.0), 0.0).astype(
            o_ref.dtype
        )

    if vmem_limit_bytes is None:
        vmem_limit_bytes = kv_vmem_bytes(seq_len, dim, q.dtype) + _VMEM_HEADROOM_BYTES
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(bh, blocks),
        in_specs=[
            pl.BlockSpec(
                (1, 1, kmax),
                lambda h, i, *_: (h * blocks + i, 0, 0),
                memory_space=pltpu.SMEM,
            ),
            pl.BlockSpec((1, block_size, dim), lambda h, i, *_: (h, i, 0)),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec((1, block_size, dim), lambda h, i, *_: (h, i, 0)),
        scratch_shapes=[
            pltpu.VMEM((seq_len, dim), k.dtype),
            pltpu.VMEM((seq_len, dim), v.dtype),
            pltpu.SemaphoreType.DMA((2,)),
        ],
    )
    out = pl.pallas_call(
        kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((bh, seq_len, dim), q.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=interpret,
    )(count, block_sizes.astype(jnp.int32), table, qh, kh, vh)
    return jnp.swapaxes(out.reshape(batch, heads, seq_len, dim), 1, 2)
