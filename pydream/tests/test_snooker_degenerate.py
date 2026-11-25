"""Tests for the snooker reference-point guard (PYDREAM_BUGS_DOCUMENTATION.md, Bug 3).

A snooker reference point z that equals the current point q0 up to round-off must not
produce a move: the direction q0 - z is noise and the Jacobian term
(d-1) * log(||q - z|| / ||q0 - z||) would make the acceptance probability ~1.
These tests do not need pysb.
"""

import numpy as np
import pytest
from scipy.stats import uniform

from pydream.Dream import Dream
from pydream.model import Model
from pydream.parameters import SampledParam

D = 36


def flat_like(x):
    return 0.0


def make_step(**kwargs):
    lower = np.zeros(D)
    param = SampledParam(uniform, loc=lower, scale=np.full(D, 3000.0))
    step = Dream(model=Model(likelihood=flat_like, sampled_parameters=[param]), **kwargs)
    step.gamma = 1.7
    return step


def mock_history(step, monkeypatch, z_list, far_pts):
    """sample_from_history returns the z points first, then the projection pairs."""
    queue = [[z] for z in z_list] + [[p] for p in far_pts]

    def fake(nseedchains, DEpairs, ndimensions, snooker=False):
        return queue.pop(0)

    monkeypatch.setattr(step, 'sample_from_history', fake)


@pytest.fixture
def points():
    rng = np.random.default_rng(1)
    q0 = rng.uniform(1, 3000, D)
    z_dup = q0 + rng.normal(0, 1e-12, D)      # stored copy of q0 after round-off
    z_far = rng.uniform(1, 3000, D)
    far = [rng.uniform(1, 3000, D) for _ in range(10)]
    return q0, z_dup, z_far, far


def test_snooker_degenerate_flag(points):
    q0, z_dup, z_far, _ = points
    step = make_step()
    assert step.snooker_degenerate(q0, z_dup)
    assert step.snooker_degenerate(q0, q0)
    assert not step.snooker_degenerate(q0, z_far)
    assert not make_step(snooker_tol=0).snooker_degenerate(q0, z_dup)
    assert make_step(snooker_tol=0).snooker_degenerate(q0, q0)


def test_degenerate_z_gives_no_move(points, monkeypatch):
    q0, z_dup, _, far = points
    step = make_step()
    mock_history(step, monkeypatch, [z_dup], far[:2])
    q, logp_prop, z = step.snooker_update(1, q0)
    np.testing.assert_array_equal(q, q0)
    # the Jacobian terms of proposal and current point cancel
    assert logp_prop == pytest.approx(np.log(np.linalg.norm(q0 - z_dup)) * (D - 1))


def test_old_behaviour_moves_with_huge_jacobian(points, monkeypatch):
    """snooker_tol = 0 reproduces the defect: a large move with log-ratio >> 0."""
    q0, z_dup, _, far = points
    step = make_step(snooker_tol=0)
    mock_history(step, monkeypatch, [z_dup], far[:2])
    q, logp_prop, z = step.snooker_update(1, q0)
    assert np.linalg.norm(q - q0) > 1.0
    jac = logp_prop - np.log(np.linalg.norm(q0 - z_dup)) * (D - 1)
    assert jac > 500


def test_regular_z_unchanged(points, monkeypatch):
    """For a regular z the guard does not change the proposal."""
    q0, _, z_far, far = points
    res = []
    for tol in (1e-8, 0):
        step = make_step(snooker_tol=tol)
        mock_history(step, monkeypatch, [z_far], far[:2])
        res.append(step.snooker_update(1, q0))
    np.testing.assert_array_equal(res[0][0], res[1][0])
    assert res[0][1] == res[1][1]
    assert np.linalg.norm(res[0][0] - q0) > 0


def test_multitry_only_degenerate_point_frozen(points, monkeypatch):
    q0, z_dup, z_far, far = points
    step = make_step()
    mock_history(step, monkeypatch, [z_dup, z_far, z_far], far[:6])
    q, logp_prop, z = step.snooker_update(3, q0)
    np.testing.assert_array_equal(q[0], q0)
    assert np.linalg.norm(q[1] - q0) > 0
    assert np.linalg.norm(q[2] - q0) > 0


# --- Bug 4: astep with an exact (or round-off) copy of q0 as snooker reference point ---

def _astep_with_z(z_offset, monkeypatch):
    """One astep with a forced snooker move whose reference point is q0 + z_offset.
    Returns (q_new, number of likelihood calls, caught warnings)."""
    import warnings

    from pydream.core import _setup_mp_dream_pool

    calls = []

    def like(x):
        calls.append(1)
        return float(-0.5 * np.sum((x - 1500.0) ** 2) / 1e6)

    lower = np.zeros(D)
    param = SampledParam(uniform, loc=lower, scale=np.full(D, 3000.0))
    step = Dream(model=Model(likelihood=like, sampled_parameters=[param]), save_history=False,
                 crossover_burnin=100, adapt_crossover=False, snooker=1.0,
                 start_random=False)
    pool = _setup_mp_dream_pool(nchains=3, niterations=10, step_instance=step)
    pool._initializer(*pool._initargs)
    pool.terminate()

    rng = np.random.default_rng(2)
    q0 = rng.uniform(1, 3000, D)
    far = [rng.uniform(1, 3000, D) for _ in range(2)]
    mock_history(step, monkeypatch, [q0 + z_offset], far)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        q_new, prior, loglike = step.astep(q0=q0, last_logprior=0.0, last_loglike=-1.0)
    return q0, q_new, len(calls), w, (prior, loglike)


@pytest.mark.parametrize('z_offset', [0.0, 1e-12])
def test_astep_frozen_snooker_no_eval_no_warning(z_offset, monkeypatch):
    q0, q_new, ncalls, w, (prior, loglike) = _astep_with_z(z_offset, monkeypatch)
    np.testing.assert_array_equal(q_new, q0)
    assert ncalls == 0                                  # likelihood not evaluated
    assert not [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert (prior, loglike) == (0.0, -1.0)              # state unchanged


def test_astep_regular_snooker_evaluates(monkeypatch):
    q0, q_new, ncalls, w, _ = _astep_with_z(500.0, monkeypatch)
    assert ncalls == 1
