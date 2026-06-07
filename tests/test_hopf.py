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


def test_value_is_running_min_over_window_not_endpoint():
    """
    TUBE semantics: V(x,t) = min over the window of l, which must be <= l at the
    single endpoint T, and must equal the min of l over a densely propagated
    trajectory. (The old single-time set returned exactly l at T; the tube
    returns the running minimum, so it can be strictly smaller.)
    """
    x0 = np.array([300.0, -200.0, 50.0, -0.5, 0.4, 0.1])

    # Endpoint-only value (the old set definition) is an UPPER bound on the tube.
    end_state = propagate_hcw(x0, (0.0, T_FINAL), np.array([T_FINAL]), N)[0]
    v_endpoint = signed_distance(end_state)
    v_tube = value(x0, t=0.0, T=T_FINAL, n=N)
    assert v_tube <= v_endpoint + 1e-6

    # Dense reference: min of l over the whole propagated trajectory.
    t_eval = np.linspace(0.0, T_FINAL, 20000)
    traj = propagate_hcw(x0, (0.0, T_FINAL), t_eval, N)
    v_reference = signed_distance(traj).min()
    np.testing.assert_allclose(v_tube, v_reference, atol=1e-2)


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


def test_boundary_points_are_on_or_inside_the_tube():
    """
    brt_boundary_points samples the preimage of the sphere AT T (the single-time
    set boundary). Under the TUBE, each such point lands exactly on the sphere
    at tau=T, so its endpoint value is 0; but its trajectory may dip INSIDE the
    sphere at some tau < T first. Hence the tube value is V <= 0 (== 0 if the
    closest approach is at T, < 0 if it passes through earlier). It is never > 0.

    This is the set-vs-tube meaning shift: the single-time set boundary is the
    OUTER surface of the tube, not the tube's own zero level set.
    """
    t = 0.0
    vel = np.array([0.1, -0.2, 0.0])
    pts = brt_boundary_points(t, T_FINAL, N, n_samples=64, velocity=vel)
    assert pts.shape == (64, 6)  # full 6D boundary states

    vals = value(pts, t=t, T=T_FINAL, n=N)
    assert np.all(vals <= 1e-6), "set-boundary points must be on/inside the tube"
    # At least some are exactly on the boundary (closest approach at T).
    assert np.any(np.abs(vals) < 1e-6)


def test_value_avoid_is_a_seam_not_implemented():
    """The controlled avoid-BRT is deferred to Step 8."""
    import pytest

    with pytest.raises(NotImplementedError):
        value_avoid(np.zeros(6), t=0.0, T=T_FINAL, n=N)


# ---------------------------------------------------------------------------
# Tube-specific tests (min-over-[t,T]). See docs/brt-tube-upgrade-spec.md.
# ---------------------------------------------------------------------------

def test_tube_catches_passthrough_that_exits_before_T():
    """
    CORE failure case the tube fixes. Construct a state whose trajectory passes
    THROUGH the sphere at some tau < T and is back OUTSIDE the sphere at exactly
    T. The single-time set (l at T) calls it SAFE; the tube must call it UNSAFE.
    """
    # Build a state at t=0 by back-propagating a state that is inside the sphere
    # at an intermediate time tau_hit and outside it at T.
    tau_hit = 0.4 * T_FINAL
    # State at tau_hit: 1 m from origin (inside R_hbr=10m), moving fast radially
    # so it is well outside the sphere shortly after.
    x_hit = np.array([1.0, 0.0, 0.0, 30.0, 0.0, 0.0])
    x0 = hcw_state_transition(N, -tau_hit) @ x_hit

    # Endpoint-only value (old set definition): outside the sphere at T.
    end_state = hcw_state_transition(N, T_FINAL) @ x0
    assert signed_distance(end_state) > 0, "fixture must be SAFE under set check"

    # Tube must see the earlier pass-through.
    assert value(x0, t=0.0, T=T_FINAL, n=N) < 0


def test_step_over_guard_fast_grazing_trajectory():
    """
    STEP-OVER guard. A fast trajectory that grazes the sphere (closest approach
    just inside R_hbr) for only a few ms must still be detected as V < 0. A naive
    coarse uniform tau grid would step over the brief passage and report SAFE;
    the bracket+refine resolution must catch it.
    """
    v_rel = 7000.0          # 7 km/s, the crossing speed
    tau_hit = 0.5 * T_FINAL
    # At tau_hit: 8 m from origin (inside 10 m), moving cross-track at 7 km/s.
    x_hit = np.array([8.0, 0.0, 0.0, 0.0, 0.0, v_rel])
    x0 = hcw_state_transition(N, -tau_hit) @ x_hit

    v = value(x0, t=0.0, T=T_FINAL, n=N)
    assert v < 0, f"fast grazing pass not detected (V={v:.3f}); tau grid stepped over"
    # The detected min distance should be ~8 m -> V ~ 8 - 10 = -2 m.
    np.testing.assert_allclose(v, 8.0 - R_HBR_DEFAULT, atol=0.5)


def test_grid_method_can_step_over_but_refine_does_not():
    """
    Demonstrates WHY refinement is required: with a deliberately coarse grid the
    'grid' method steps over the ms-long fast passage (reports SAFE), while the
    default 'refine' method detects it. Guards against silently regressing to a
    grid-only implementation.
    """
    v_rel = 7000.0
    coarse = 21  # ~30 s spacing -> far coarser than the ~ms dwell
    # Place the hit deliberately BETWEEN two coarse nodes (off-grid) so the grid
    # genuinely steps over it. Nodes sit at multiples of T/(coarse-1); 0.537 is
    # not such a multiple.
    tau_hit = 0.537 * T_FINAL
    x_hit = np.array([5.0, 0.0, 0.0, 0.0, 0.0, v_rel])
    x0 = hcw_state_transition(N, -tau_hit) @ x_hit

    v_grid = value(x0, t=0.0, T=T_FINAL, n=N, n_time=coarse, method="grid")
    v_refine = value(x0, t=0.0, T=T_FINAL, n=N, n_time=coarse, method="refine")
    assert v_grid > 0, "coarse grid should step over (this is the hazard)"
    assert v_refine < 0, "refine must catch the passage the coarse grid missed"


def test_monotonicity_along_trajectory():
    """
    V(x(t), t) is non-decreasing in t along a propagated trajectory: as t grows
    the min is over a shorter remaining window, so remaining risk only shrinks
    (signed distance only rises). This is what makes t* well defined.
    """
    x0 = np.array([400.0, -300.0, 80.0, -1.0, 0.6, 0.2])
    t_samples = np.linspace(0.0, T_FINAL, 12)
    states = propagate_hcw(x0, (0.0, T_FINAL), t_samples, N)
    vals = np.array([
        value(states[i], t=t_samples[i], T=T_FINAL, n=N)
        for i in range(len(t_samples))
    ])
    diffs = np.diff(vals)
    assert np.all(diffs >= -1e-3), f"V not non-decreasing in t: diffs={diffs}"


def test_batch_time_axis_shape_and_consistency():
    """Vectorized (N,6) tube values match per-row scalar calls (new time axis)."""
    rng = np.random.default_rng(7)
    X = rng.normal(scale=200.0, size=(9, 6))
    batched = value(X, t=0.0, T=T_FINAL, n=N)
    assert batched.shape == (9,)
    for i in range(X.shape[0]):
        np.testing.assert_allclose(
            batched[i], value(X[i], t=0.0, T=T_FINAL, n=N), atol=1e-3
        )


def test_nominal_crossing_closest_approach_is_pinned_by_refine():
    """
    Pin the default resolution against KNOWN physics: build a crossing whose
    closest approach is a prescribed miss distance at a known time, and confirm
    the default (auto-sized grid + refine) recovers that miss distance. This
    validates the n_time default empirically rather than by gut.
    """
    miss = 4.0               # m, inside R_hbr=10m -> should be unsafe
    v_rel = 7000.0
    tau_hit = 0.5 * T_FINAL
    x_hit = np.array([miss, 0.0, 0.0, 0.0, 0.0, v_rel])
    x0 = hcw_state_transition(N, -tau_hit) @ x_hit

    v = value(x0, t=0.0, T=T_FINAL, n=N)  # default n_time, default refine
    np.testing.assert_allclose(v, miss - R_HBR_DEFAULT, atol=0.3)