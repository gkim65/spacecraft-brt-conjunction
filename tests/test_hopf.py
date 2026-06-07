"""
Tests for the uncontrolled backward-reachable tube (reachability/hopf.py).

Guards the two things easiest to get silently wrong:
  1. Sign convention -- V must inherit target_set's l(x) convention exactly,
     and reduce to l(x) at the final time t = T.
  2. Time direction -- the forward flow uses dt = T - t >= 0; a flipped sign
     would invert the tube without raising an error.
"""

import numpy as np

from spacecraft_brt.dynamics.hcw import (
    hcw_state_transition,
    orbital_rate,
    propagate_hcw,
)
from spacecraft_brt.reachability.hopf import brt_boundary_points, value, value_avoid
from spacecraft_brt.reachability.target_set import signed_distance
from spacecraft_brt.utils.constants import R_EARTH, R_HBR_DEFAULT

# A representative LEO encounter setup.
SMA = R_EARTH + 700e3
N = orbital_rate(SMA)
T_FINAL = 600.0  # 10-minute horizon (s)


def test_value_reduces_to_target_set_at_final_time():
    """CRITICAL: V(x, T) == l(x) exactly, since Phi(0) = I."""
    rng = np.random.default_rng(0)
    x = rng.normal(scale=50.0, size=6)
    np.testing.assert_allclose(
        value(x, t=T_FINAL, T=T_FINAL, n=N),
        signed_distance(x),
        rtol=0,
        atol=1e-12,
    )


def test_state_inside_sphere_at_T_is_inside_brt():
    """A state already inside the collision sphere at t=T has V < 0."""
    x = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 2 m < R_hbr
    assert value(x, t=T_FINAL, T=T_FINAL, n=N) < 0


def test_far_static_state_is_outside_brt():
    """A far state with no velocity stays far -> V > 0 over the horizon."""
    x = np.array([5000.0, 5000.0, 0.0, 0.0, 0.0, 0.0])
    assert value(x, t=0.0, T=T_FINAL, n=N) > 0


def test_collision_course_state_is_inside_brt():
    """
    A state that is FAR at t=0 but flows INTO the sphere by T must be inside the
    BRT (V < 0), even though l(x) at t=0 is large and positive. This is the
    point of the BRT: membership is decided by the future, not the present.
    """
    # Final state inside the sphere, with nonzero velocity so its back-image is
    # genuinely far away. Search a few velocities for one that is far at t=0.
    phi_back = hcw_state_transition(N, -T_FINAL)  # Phi(0 - T)
    x0 = None
    for v in (50.0, 20.0, 10.0, 5.0, 2.0):
        x_T = np.array([1.0, 0.0, 0.0, v, 0.0, 0.0])  # inside sphere at T
        candidate = phi_back @ x_T
        if signed_distance(candidate) > 10.0 * R_HBR_DEFAULT:
            x0 = candidate
            break
    assert x0 is not None, "no far-at-t=0 collision-course fixture found"

    assert signed_distance(x0) > R_HBR_DEFAULT   # genuinely far at t=0
    assert value(x0, t=0.0, T=T_FINAL, n=N) < 0  # but inside the BRT


def test_value_matches_explicit_forward_propagation():
    """V(x,t) must equal l(x) evaluated at the forward-propagated state."""
    x0 = np.array([300.0, -200.0, 50.0, -0.5, 0.4, 0.1])
    states = propagate_hcw(x0, (0.0, T_FINAL), np.array([T_FINAL]), N)
    expected = signed_distance(states[0])
    np.testing.assert_allclose(value(x0, t=0.0, T=T_FINAL, n=N), expected, atol=1e-6)


def test_time_direction_phi_composition_is_identity():
    """Phi(T-t) @ Phi(t-T) == I -- the forward/backward flow must invert."""
    t = 120.0
    fwd = hcw_state_transition(N, T_FINAL - t)
    bwd = hcw_state_transition(N, t - T_FINAL)
    np.testing.assert_allclose(fwd @ bwd, np.eye(6), atol=1e-9)


def test_vectorized_matches_scalar():
    """Batched (N,6) input must match per-row scalar calls."""
    rng = np.random.default_rng(42)
    X = rng.normal(scale=100.0, size=(7, 6))
    batched = value(X, t=0.0, T=T_FINAL, n=N)
    assert batched.shape == (7,)
    for i in range(X.shape[0]):
        np.testing.assert_allclose(batched[i], value(X[i], t=0.0, T=T_FINAL, n=N))


def test_boundary_points_have_zero_value():
    """Sampled boundary points must satisfy V ~ 0 (they map onto the sphere)."""
    t = 0.0
    vel = np.array([0.1, -0.2, 0.0])
    pts = brt_boundary_points(t, T_FINAL, N, n_samples=64, velocity=vel)
    assert pts.shape == (64, 6)  # full 6D boundary states

    # Every returned boundary state must lie on the BRT boundary (V == 0).
    vals = value(pts, t=t, T=T_FINAL, n=N)
    np.testing.assert_allclose(vals, 0.0, atol=1e-6)


def test_value_avoid_is_a_seam_not_implemented():
    """The controlled avoid-BRT is deferred to Step 8."""
    import pytest

    with pytest.raises(NotImplementedError):
        value_avoid(np.zeros(6), t=0.0, T=T_FINAL, n=N)