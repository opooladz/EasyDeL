# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Numerical-precision regression tests for the chunked distillation KL.

Guards against a class of bf16 precision defects in the chunked loss where the
logged ``kl_loss`` could read *negative* (impossible for a forward KL) and
carried an O(1e-2) error floor at large vocabularies:

* Under ``jax.jit``, the teacher and student log-sum-exp reductions lower to
  *separate* fused reduce ops. With a bf16 accumulator their order-dependent
  roundings differ, so the error does not cancel in ``lse_t - lse_s`` — even on
  bit-identical inputs. The fix accumulates both reductions in fp32 (the
  upcast is fused into the reduction; no fp32 ``[..., V]`` buffer exists).
* The ``distill_xent`` / ``teacher_entropy`` diagnostics subtract O(|logits|)
  quantities to produce an O(1) result and previously used bf16-normalised
  teacher probabilities (``sum(p_t) = 1 + O(1e-2)``), leaking
  ``O(delta * |logits|)`` errors into the readings.

The decisive invariance: for identical student and teacher logits the per-token
KL must be ~0 under jit. Pre-fix, at production-like vocab/peakedness, the jit
value was ~+2.5e-2 on identical inputs (and sign-indefinite across backends).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from easydel.trainers.distillation_trainer._fn import chunked_distillation_loss

# Peaked logits at magnitudes where bf16 spacing is coarse (|x| ~ 30-60 -> ulp
# 0.25-0.5): the regime where the reduction-rounding defect is visible.
_VOCAB = 8192
_BATCH = 2
_SEQ = 512
_CHUNK = 128  # multiple chunks -> exercises the scan accumulation
_SCALE = 12.0

_IDENTITY = lambda h: h  # noqa: E731  hidden==logits; isolates the loss arithmetic


def _make_logits(seed: int, noise: float) -> tuple[jax.Array, jax.Array]:
    k_t, k_n = jax.random.split(jax.random.PRNGKey(seed))
    teacher = jax.random.normal(k_t, (_BATCH, _SEQ, _VOCAB), jnp.float32) * _SCALE
    student = teacher + noise * jax.random.normal(k_n, (_BATCH, _SEQ, _VOCAB), jnp.float32)
    return teacher.astype(jnp.bfloat16), student.astype(jnp.bfloat16)


def _run_chunked(student: jax.Array, teacher: jax.Array) -> dict[str, jax.Array]:
    mask = jnp.ones((_BATCH, _SEQ), jnp.bfloat16)

    @jax.jit
    def step(s, t):
        _, metrics = chunked_distillation_loss(
            student_hidden=s,
            teacher_hidden=t,
            student_lm_head_fn=_IDENTITY,
            teacher_lm_head_fn=_IDENTITY,
            attention_mask=mask,
            labels=None,
            use_hard_labels=False,
            temperature=1.0,
            alpha=1.0,
            chunk_size=_CHUNK,
            checkpoint_chunks=True,
        )
        return metrics

    return step(student, teacher)


def _fp32_reference(student: jax.Array, teacher: jax.Array) -> tuple[float, float, float]:
    t = teacher.astype(jnp.float32)
    s = student.astype(jnp.float32)
    lp_t = jax.nn.log_softmax(t, axis=-1)
    lp_s = jax.nn.log_softmax(s, axis=-1)
    p_t = jnp.exp(lp_t)
    kl = float(jnp.mean(jnp.sum(p_t * (lp_t - lp_s), axis=-1)))
    xent = float(jnp.mean(-jnp.sum(p_t * lp_s, axis=-1)))
    entropy = float(jnp.mean(-jnp.sum(p_t * lp_t, axis=-1)))
    return kl, xent, entropy


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kl_exactly_zero_for_identical_logits_under_jit(seed):
    """KL(x, x) must vanish under jit; pre-fix it read O(1e-2), sign-indefinite."""
    teacher, _ = _make_logits(seed, noise=0.0)
    metrics = _run_chunked(teacher, teacher)
    kl = float(metrics["kl_loss"])
    assert abs(kl) <= 1e-5, f"KL on identical inputs should be ~0 under jit, got {kl}"


@pytest.mark.parametrize("noise,atol", [(0.15, 3e-3), (0.6, 1.5e-2)])
def test_kl_tracks_fp32_reference(noise, atol):
    """Measured kl_loss matches an fp32 dense reference on the same bf16 logits."""
    teacher, student = _make_logits(7, noise=noise)
    ref_kl, _, _ = _fp32_reference(student, teacher)
    kl = float(_run_chunked(student, teacher)["kl_loss"])
    assert kl == pytest.approx(ref_kl, abs=atol)
    assert kl >= -1e-5, f"forward KL must be non-negative, got {kl}"


def test_xent_and_entropy_metrics_match_reference():
    """The diagnostic split is accurate and satisfies xent - entropy ~= kl."""
    teacher, student = _make_logits(11, noise=0.6)
    ref_kl, ref_xent, ref_entropy = _fp32_reference(student, teacher)
    metrics = _run_chunked(student, teacher)
    xent = float(metrics["distill_xent_loss"])
    entropy = float(metrics["teacher_entropy_loss"])
    kl = float(metrics["kl_loss"])
    # Pre-fix, xent could read negative (e.g. -1.35 for a true +0.97) via the
    # bf16-normalised-probability mass defect.
    assert xent == pytest.approx(ref_xent, rel=0.02)
    assert entropy == pytest.approx(ref_entropy, rel=0.02)
    assert (xent - entropy) == pytest.approx(kl, abs=5e-2)


def test_gradient_matches_analytic_kd_gradient():
    """d(loss)/d(student) ~= (softmax(s) - softmax(t)) / n_tokens (unchanged by the fix)."""
    teacher, student = _make_logits(3, noise=0.4)
    mask = jnp.ones((_BATCH, _SEQ), jnp.bfloat16)

    def loss_of(s):
        total, _ = chunked_distillation_loss(
            student_hidden=s,
            teacher_hidden=teacher,
            student_lm_head_fn=_IDENTITY,
            teacher_lm_head_fn=_IDENTITY,
            attention_mask=mask,
            labels=None,
            use_hard_labels=False,
            temperature=1.0,
            alpha=1.0,
            chunk_size=_CHUNK,
            checkpoint_chunks=True,
        )
        return total.astype(jnp.float32)

    grad = jax.jit(jax.grad(loss_of))(student).astype(jnp.float32)
    analytic = (
        jax.nn.softmax(student.astype(jnp.float32), axis=-1) - jax.nn.softmax(teacher.astype(jnp.float32), axis=-1)
    ) / (_BATCH * _SEQ)
    max_err = float(jnp.max(jnp.abs(grad - analytic)))
    max_ref = float(jnp.max(jnp.abs(analytic)))
    # bf16 softmax tolerance: errors are O(ulp) of the probabilities.
    assert max_err <= 0.35 * max_ref + 1e-6
