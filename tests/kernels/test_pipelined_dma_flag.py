# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

import inspect
import os
import sys
from unittest.mock import MagicMock, patch

import jax
import jax.extend.core
import jax.numpy as jnp
import numpy as np
import pytest

# Provide lightweight stubs for optional heavy runtime dependencies if not present
for mod_name in ("torch", "vllm", "vllm.utils"):
    if mod_name not in sys.modules:
        try:
            __import__(mod_name)
        except ImportError:
            sys.modules[mod_name] = MagicMock()

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.kernels.ragged_paged_attention.v2 import (  # noqa: E402
    ragged_kv_cache_update,
)
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel  # noqa: E402


def test_env_var_default():
    """Verify that both flags default to False when unset."""
    with patch.dict(os.environ, {}, clear=True):
        assert envs.USE_RPA_PIPELINED_DMA_STAGING is False
        assert envs.USE_PIPELINED_DMA_STAGING is False


def test_env_var_toggle():
    """Verify that setting either flag toggles both flags."""
    # Test setting USE_RPA_PIPELINED_DMA_STAGING="1"
    with patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "1"}, clear=True):
        assert envs.USE_RPA_PIPELINED_DMA_STAGING is True
        assert envs.USE_PIPELINED_DMA_STAGING is True

    # Test setting USE_PIPELINED_DMA_STAGING="1"
    with patch.dict(os.environ, {"USE_PIPELINED_DMA_STAGING": "1"}, clear=True):
        assert envs.USE_RPA_PIPELINED_DMA_STAGING is True
        assert envs.USE_PIPELINED_DMA_STAGING is True

    # Test boolean string true
    with patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "true"}, clear=True):
        assert envs.USE_RPA_PIPELINED_DMA_STAGING is True
        assert envs.USE_PIPELINED_DMA_STAGING is True

    # Test explicit disable "0"
    with patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "0"}, clear=True):
        assert envs.USE_RPA_PIPELINED_DMA_STAGING is False
        assert envs.USE_PIPELINED_DMA_STAGING is False


def test_ragged_kv_cache_update_signature_and_wiring():
    """Verify ragged_kv_cache_update parameter defaults and env resolution."""
    sig = inspect.signature(ragged_kv_cache_update.kv_cache_update)
    assert "pipelined_dma" in sig.parameters
    assert sig.parameters["pipelined_dma"].default is None

    mock_kv = MagicMock()
    mock_new_kv = MagicMock()
    mock_new_kv.shape = (10, 2, 128)
    unwrapped_update = inspect.unwrap(ragged_kv_cache_update.kv_cache_update)

    # Verify when env var is False
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(ragged_kv_cache_update, "_kv_cache_update") as mock_update,
    ):
        unwrapped_update(
            mock_new_kv, MagicMock(), mock_kv, MagicMock(), num_slices_per_block=4
        )
        mock_update.assert_called_once()
        _, kwargs = mock_update.call_args
        assert kwargs["pipelined_dma"] is False

    # Verify when env var is True
    with (
        patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "1"}, clear=True),
        patch.object(ragged_kv_cache_update, "_kv_cache_update") as mock_update,
    ):
        unwrapped_update(
            mock_new_kv, MagicMock(), mock_kv, MagicMock(), num_slices_per_block=4
        )
        mock_update.assert_called_once()
        _, kwargs = mock_update.call_args
        assert kwargs["pipelined_dma"] is True

    # Verify explicit override pipelined_dma=False when env var is True
    with (
        patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "1"}, clear=True),
        patch.object(ragged_kv_cache_update, "_kv_cache_update") as mock_update,
    ):
        unwrapped_update(
            mock_new_kv,
            MagicMock(),
            mock_kv,
            MagicMock(),
            num_slices_per_block=4,
            pipelined_dma=False,
        )
        mock_update.assert_called_once()
        _, kwargs = mock_update.call_args
        assert kwargs["pipelined_dma"] is False


def test_kv_cache_update_builds_block_specs_and_traces():
    """Verify _kv_cache_update builds pl.ANY BlockSpecs and traces cleanly."""
    import functools

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    new_kv = jnp.zeros((16, 2, 128), dtype=jnp.bfloat16)
    kv_cache = jnp.zeros((64, 2, 128), dtype=jnp.bfloat16)
    slices = jnp.zeros((3, 4), dtype=jnp.int32)
    num_slices = jnp.array([2], dtype=jnp.int32)

    for pipelined_dma in (False, True):
        with patch.object(
            ragged_kv_cache_update.pl,
            "pallas_call",
            wraps=ragged_kv_cache_update.pl.pallas_call,
        ) as spy_pallas_call:
            out_shape = jax.eval_shape(
                functools.partial(
                    ragged_kv_cache_update.kv_cache_update,
                    num_slices_per_block=4,
                    pipelined_dma=pipelined_dma,
                ),
                new_kv,
                slices,
                kv_cache,
                num_slices,
            )
            assert out_shape.shape == kv_cache.shape
            assert out_shape.dtype == kv_cache.dtype
            spy_pallas_call.assert_called_once()
            _, call_kwargs = spy_pallas_call.call_args
            grid_spec = call_kwargs["grid_spec"]
            assert [spec.memory_space for spec in grid_spec.in_specs] == [
                pl.ANY,
                pl.ANY,
            ]
            assert [spec.memory_space for spec in grid_spec.out_specs] == [pl.ANY]


def test_kernel_v3_prepare_inputs_outputs_wiring():
    """Verify kernel.py prepare_inputs/prepare_outputs wiring to env flag."""
    sig_in = inspect.signature(kernel.prepare_inputs)
    assert "fuse_non_tiling_axis_swap" in sig_in.parameters
    assert sig_in.parameters["fuse_non_tiling_axis_swap"].default is None

    sig_out = inspect.signature(kernel.prepare_outputs)
    assert "fuse_non_tiling_axis_swap" in sig_out.parameters
    assert sig_out.parameters["fuse_non_tiling_axis_swap"].default is None


def test_kernel_v3_prepare_inputs_outputs_execution():
    """Verify prepare_inputs and prepare_outputs shape behavior under env toggle."""
    import jax.numpy as jnp

    max_tokens = 16
    q_heads = 8
    kv_heads = 2
    head_dim = 128
    q = jnp.zeros((max_tokens, q_heads, head_dim))
    k = jnp.zeros((max_tokens, kv_heads, head_dim))
    v = jnp.zeros((max_tokens, kv_heads, head_dim))

    # Default / False: returns [kv_heads, max_tokens, q_heads_per_kv_head, 1, head_dim]
    with patch.dict(os.environ, {}, clear=True):
        q_out, _ = kernel.prepare_inputs(q, k, v)
        assert q_out.shape == (2, 16, 4, 1, 128)
        out = kernel.prepare_outputs(q_out, 4, 128)
        assert out.shape == (16, 8, 128)

    # Flag enabled: returns [max_tokens, kv_heads, q_heads_per_kv_head, 1, head_dim]
    with patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "1"}, clear=True):
        q_out, _ = kernel.prepare_inputs(q, k, v)
        assert q_out.shape == (16, 2, 4, 1, 128)
        out = kernel.prepare_outputs(q_out, 4, 128)
        assert out.shape == (16, 8, 128)

    # Explicit override False when flag is True
    with patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "1"}, clear=True):
        q_out, _ = kernel.prepare_inputs(q, k, v, fuse_non_tiling_axis_swap=False)
        assert q_out.shape == (2, 16, 4, 1, 128)
        out = kernel.prepare_outputs(q_out, 4, 128, fuse_non_tiling_axis_swap=False)
        assert out.shape == (16, 8, 128)


@pytest.mark.parametrize("fuse", [None, True, False])
@pytest.mark.parametrize("env", ["0", "1"])
@pytest.mark.parametrize(
    "public, jitted",
    [
        ("ragged_paged_attention", "_ragged_paged_attention"),
        ("ragged_paged_attention_bundled", "_ragged_paged_attention_bundled"),
    ],
)
def test_kernel_v3_entry_points_resolve_flag_before_jit(public, jitted, env, fuse):
    """None reads USE_RPA_PIPELINED_DMA_STAGING outside the jit (R-4.3); an
    explicit value wins over the env."""
    with (
        patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": env}, clear=True),
        patch.object(kernel, jitted) as mock_jitted,
    ):
        getattr(kernel, public)("q", sm_scale=0.5, fuse_non_tiling_axis_swap=fuse)
    expected = env == "1" if fuse is None else fuse
    mock_jitted.assert_called_once_with(
        "q", sm_scale=0.5, fuse_non_tiling_axis_swap=expected
    )


def _primitives_outside_kernels(jaxpr, found):
    for eqn in jaxpr.eqns:
        found.append(eqn.primitive.name)
        if eqn.primitive.name == "pallas_call":
            continue
        for sub in jax.extend.core.jaxprs_in_params(eqn.params):
            _primitives_outside_kernels(sub, found)
    return found


@pytest.mark.parametrize("kv_dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
@pytest.mark.parametrize("num_q_heads, num_kv_heads", [(8, 2), (4, 4), (16, 1)])
def test_fused_axis_swap_traces_no_transposes(
    monkeypatch, num_q_heads, num_kv_heads, kv_dtype
):
    """With the swap fused into the kernel DMAs, XLA no longer transposes the
    queries before RPA v3 or the outputs after it. Traced on CPU, so every
    decode, prefill and mixed kernel body is built, DMA descriptors included."""
    monkeypatch.setattr(kernel, "get_tpu_version", lambda: 7)
    max_num_tokens, head_dim, page_size, max_num_seqs, pages_per_seq = 64, 128, 16, 4, 4
    args = (
        jnp.zeros((max_num_tokens, num_q_heads, head_dim), jnp.bfloat16),
        jnp.zeros((max_num_tokens, num_kv_heads, head_dim), kv_dtype),
        jnp.zeros((max_num_tokens, num_kv_heads, head_dim), kv_dtype),
        jnp.zeros(
            kernel.get_kv_cache_shape(32, page_size, num_kv_heads, head_dim, kv_dtype),
            kv_dtype,
        ),
        jnp.zeros((max_num_seqs,), jnp.int32),
        jnp.zeros((max_num_seqs * pages_per_seq,), jnp.int32),
        jnp.zeros((max_num_seqs + 1,), jnp.int32),
        jnp.zeros((3,), jnp.int32),
    )

    def transposes(fuse):
        jaxpr = jax.make_jaxpr(
            lambda *a: kernel.ragged_paged_attention(
                *a,
                sm_scale=head_dim**-0.5,
                chunk_prefill_size=16,
                vmem_limit_bytes=64 * 1024 * 1024,
                fuse_non_tiling_axis_swap=fuse,
            )
        )(*args)
        found = _primitives_outside_kernels(jaxpr.jaxpr, [])
        assert found.count("pallas_call") == 3  # decode, prefill, mixed
        assert jaxpr.out_avals[0].shape == args[0].shape
        return found.count("transpose")

    assert transposes(fuse=False) == 2
    assert transposes(fuse=True) == 0


# (q_lens, kv_lens, distribution, chunk_prefill_size) per batch kind. Uneven
# lengths leave partial query blocks and blocks that start mid-tile.
_BATCHES = {
    "decode": ([1] * 8, [5, 17, 33, 64, 1, 90, 128, 47], (8, 8, 8), None),
    "prefill": ([37, 150, 5, 64], [37, 150, 5, 64], (0, 0, 4), None),
    "chunked": ([32, 32, 32], [32, 75, 160], (0, 3, 3), 32),
    "mixed": (
        [1, 1, 1, 32, 32, 20, 100],
        [40, 9, 128, 64, 96, 20, 150],
        (3, 5, 7),
        32,
    ),
}


def _rpa_batch(batch, num_q_heads, num_kv_heads, kv_dtype, seed=0):
    q_lens, kv_lens, distribution, chunk_prefill_size = _BATCHES[batch]
    max_num_tokens, max_num_seqs, head_dim, page_size = 320, 8, 128, 16
    pages_per_seq = -(-max(kv_lens) // page_size)
    total_num_pages = max_num_seqs * pages_per_seq + 8
    num_seqs = len(q_lens)
    rng = np.random.default_rng(seed)

    def normal(*shape):
        return rng.standard_normal(shape, dtype=np.float32)

    cu_q_lens = np.zeros(max_num_seqs + 1, np.int32)
    cu_q_lens[1 : num_seqs + 1] = np.cumsum(q_lens)
    cu_q_lens[num_seqs + 1 :] = cu_q_lens[num_seqs]
    return dict(
        q=normal(max_num_tokens, num_q_heads, head_dim),
        k=normal(max_num_tokens, num_kv_heads, head_dim),
        v=normal(max_num_tokens, num_kv_heads, head_dim),
        kv_cache=normal(
            *kernel.get_kv_cache_shape(
                total_num_pages, page_size, num_kv_heads, head_dim, kv_dtype
            )
        ),
        kv_lens=np.pad(np.asarray(kv_lens, np.int32), (0, max_num_seqs - num_seqs)),
        page_indices=rng.permutation(total_num_pages)[
            : max_num_seqs * pages_per_seq
        ].astype(np.int32),
        cu_q_lens=cu_q_lens,
        distribution=np.asarray(distribution, np.int32),
        chunk_prefill_size=chunk_prefill_size,
        num_tokens=int(cu_q_lens[-1]),
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("kv_dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
@pytest.mark.parametrize("num_q_heads, num_kv_heads", [(8, 2), (4, 4), (16, 1), (6, 2)])
@pytest.mark.parametrize("batch", list(_BATCHES))
def test_fused_axis_swap_matches_unfused_and_reference(
    batch, num_q_heads, num_kv_heads, kv_dtype
):
    """Fusing the query/output axis swap into the RPA v3 DMAs changes only how
    blocks move between HBM and VMEM, so outputs and caches stay bit-identical
    to the unfused path, and both match the reference."""
    if jax.default_backend() != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {jax.default_backend()}.")
    b = _rpa_batch(batch, num_q_heads, num_kv_heads, kv_dtype)
    n = b["num_tokens"]
    sm_scale = 128**-0.5
    scales = {} if kv_dtype == jnp.bfloat16 else dict(k_scale=0.5, v_scale=0.5)

    def inputs():
        # Fresh device arrays per call: the kernel donates them.
        return (
            jnp.asarray(b["q"], jnp.bfloat16),
            jnp.asarray(b["k"], kv_dtype),
            jnp.asarray(b["v"], kv_dtype),
            jnp.asarray(b["kv_cache"], kv_dtype),
            jnp.asarray(b["kv_lens"]),
            jnp.asarray(b["page_indices"]),
            jnp.asarray(b["cu_q_lens"]),
            jnp.asarray(b["distribution"]),
        )

    def as_f32(x):
        return np.asarray(x.astype(jnp.float32))

    results = {}
    for fuse in (False, True):
        out, cache = kernel.ragged_paged_attention(
            *inputs(),
            sm_scale=sm_scale,
            chunk_prefill_size=b["chunk_prefill_size"],
            fuse_non_tiling_axis_swap=fuse,
            **scales,
        )
        results[fuse] = (as_f32(out)[:n], as_f32(cache))
    ref_out, ref_cache = kernel.ref_ragged_paged_attention(
        *inputs(), sm_scale=sm_scale, **scales
    )

    np.testing.assert_array_equal(results[True][0], results[False][0])
    np.testing.assert_array_equal(results[True][1], results[False][1])
    for out, cache in results.values():
        np.testing.assert_allclose(out, as_f32(ref_out), atol=0.2, rtol=0.2)
        np.testing.assert_array_equal(cache, as_f32(ref_cache))
