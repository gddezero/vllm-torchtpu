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
"""Numerics tests for the explicit block-sparse attention Pallas kernel.

On TPU the kernel is compiled; elsewhere it runs in Pallas TPU interpret mode,
so the same cases check the kernel logic on CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.block_sparse_attention.kernel import (
    block_sparse_attention_bshd,
    block_tables,
    is_supported,
)

def _interpret():
    # Decided per call: touching the backend at import time claims the TPU
    # during collection (see tests/conftest.py).
    return False if jax.default_backend() == "tpu" else pltpu.InterpretParams()


def _reference(q, k, v, block_map, block_sizes, block_size, scale):
    """f32 softmax over the keys each query tile selects; zero rows when none."""
    seq_len = q.shape[1]
    tile = np.arange(seq_len) // block_size
    key_valid = (np.arange(seq_len) % block_size) < np.asarray(block_sizes)[tile]
    sel = np.asarray(block_map)[:, :, tile][:, :, :, tile]  # (B, H, S, S)
    mask = jnp.asarray(sel & key_valid[None, None, None, :])
    s = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
    has_key = mask.any(axis=-1, keepdims=True)
    p = jax.nn.softmax(jnp.where(mask, s, -1e30), axis=-1)
    p = jnp.where(mask & has_key, p, 0.0)
    return jnp.einsum("bhqk,bkhd->bqhd", p, v.astype(jnp.float32))


def _inputs(batch, blocks, heads, dim, block_size, seed):
    kq, kk, kv, km = jax.random.split(jax.random.PRNGKey(seed), 4)
    shape = (batch, blocks * block_size, heads, dim)
    q, k, v = (jax.random.normal(key, shape, jnp.float32).astype(jnp.bfloat16) for key in (kq, kk, kv))
    block_map = np.array(jax.random.bernoulli(km, 0.35, (batch, heads, blocks, blocks)))
    return q, k, v, block_map


def _rel_err(out, ref):
    out = np.asarray(out, np.float32)
    ref = np.asarray(ref, np.float32)
    assert out.shape == ref.shape
    assert np.all(np.isfinite(out))
    return np.abs(out - ref).max() / np.abs(ref).max()


def _run(q, k, v, block_map, sizes, block_size, kb):
    return block_sparse_attention_bshd(
        q,
        k,
        v,
        jnp.asarray(block_map),
        jnp.asarray(sizes),
        scale=q.shape[-1] ** -0.5,
        block_size=block_size,
        kb=kb,
        interpret=_interpret(),
    )


@pytest.mark.parametrize(
    "batch,blocks,heads,kb",
    [
        (1, 5, 2, 4),  # kb does not divide the tile count
        (1, 8, 3, 2),
        (2, 3, 2, 4),  # batch folded with heads; kmax > blocks
        (1, 6, 1, 1),
    ],
)
def test_matches_masked_reference(batch, blocks, heads, kb):
    block_size, dim = 128, 128
    q, k, v, block_map = _inputs(batch, blocks, heads, dim, block_size, seed=blocks + 10 * heads)
    sizes = np.full(blocks, block_size, np.int32)
    sizes[-1] = 37  # ragged tail
    sizes[1] = 0  # a selected tile with no valid rows
    block_map[0, 0, 0, :] = False  # a query tile with no selection -> zeros
    block_map[0, -1, -1, :] = False
    block_map[0, -1, -1, 1] = True  # selects only the empty tile -> zeros
    block_map[0, 0, 2, :] = True  # a dense query tile
    out = _run(q, k, v, block_map, sizes, block_size, kb)
    ref = _reference(q, k, v, block_map, sizes, block_size, q.shape[-1] ** -0.5)
    assert _rel_err(out, ref) < 2e-2
    np.testing.assert_array_equal(np.asarray(out[0, :block_size, 0], np.float32), 0.0)
    np.testing.assert_array_equal(np.asarray(out[0, -block_size:, -1], np.float32), 0.0)


def test_detects_a_flipped_block():
    """Negative control: the tolerance above must see a one-tile difference."""
    block_size, dim, blocks, heads = 128, 128, 5, 2
    q, k, v, block_map = _inputs(1, blocks, heads, dim, block_size, seed=7)
    sizes = np.full(blocks, block_size, np.int32)
    block_map[0, 1, 3, :] = False
    block_map[0, 1, 3, [0, 2]] = True
    flipped = block_map.copy()
    flipped[0, 1, 3, 4] = True
    out = _run(q, k, v, flipped, sizes, block_size, 4)
    ref = _reference(q, k, v, block_map, sizes, block_size, dim**-0.5)
    assert _rel_err(out, ref) > 0.1


def test_block_tables_sorted_and_counted():
    block_map = np.zeros((1, 1, 2, 5), bool)
    block_map[0, 0, 0, [4, 1, 3]] = True
    table, count = block_tables(jnp.asarray(block_map), kb=4)
    assert table.shape == (2, 1, 8)
    np.testing.assert_array_equal(np.asarray(count), [3, 0])
    np.testing.assert_array_equal(np.asarray(table[0, 0]), [1, 3, 4, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(np.asarray(table[1, 0]), 0)


def test_is_supported_vmem_envelope():
    mib = 1024 * 1024
    assert is_supported(128 * 800, 128, jnp.bfloat16, 128, 128 * mib)  # 2 x 25 MiB K/V
    assert not is_supported(128 * 2000, 128, jnp.bfloat16, 128, 128 * mib)  # 2 x 62.5 MiB
    assert not is_supported(128 * 4, 64, jnp.bfloat16, 128, 128 * mib)  # head_dim not lane aligned
    assert not is_supported(100, 128, jnp.bfloat16, 128, 128 * mib)  # not whole tiles
