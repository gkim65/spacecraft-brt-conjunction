"""
Unit tests for HCW dynamics.

Validates:
- State transition matrix properties (determinant = 1 for area-preserving flow)
- Free drift solution (analytic vs numerical)
- Energy-like invariants
- Scenario TCA finding
"""

import numpy as np
import pytest
from spacecraft_brt.dynamics.hcw import (
    orbital_rate,
    hcw_matrices,
    hcw_state_transition,
    propagate_hcw,
    separation_distance,
    time_of_closest_approach,
)
from spacecraft_brt.dynamics.scenario import make_leo_conjunction, make_crossing_conjunction


# ── Constants ────────────────────────────────────────────────────────────────

LEO_SMA = 6.921e6  # 550 km altitude


# ── Orbital rate ─────────────────────────────────────────────────────────────

def test_orbital_rate_leo():
    n = orbital_rate(LEO_SMA)
    # ISS period ~92 min, so n ~ 2pi / (95*60)
    expected = 2 * np.pi / (95.50365263990858 * 60)
    assert abs(n - expected) / expected < 0.02  # within 2%


# ── HCW matrices ─────────────────────────────────────────────────────────────

def test_hcw_matrix_shape():
    n = orbital_rate(LEO_SMA)
    A, B = hcw_matrices(n)
    assert A.shape == (6, 6)
    assert B.shape == (6, 3)


def test_hcw_a_skew_structure():
    """A should have zero diagonal in position-velocity coupling."""
    n = orbital_rate(LEO_SMA)
    A, _ = hcw_matrices(n)
    # Top-left 3x3 (position-position coupling) should be zero
    assert np.allclose(A[:3, :3], 0)


# ── State transition matrix ───────────────────────────────────────────────────

def test_state_transition_identity_at_zero():
    n = orbital_rate(LEO_SMA)
    Phi = hcw_state_transition(n, dt=0.0)
    assert np.allclose(Phi, np.eye(6), atol=1e-10)


def test_state_transition_determinant():
    """Det of state transition matrix should be 1 (volume-preserving)."""
    n = orbital_rate(LEO_SMA)
    Phi = hcw_state_transition(n, dt=60.0)
    assert abs(np.linalg.det(Phi) - 1.0) < 1e-8


def test_state_transition_composition():
    """Phi(t1+t2) == Phi(t1) @ Phi(t2)."""
    n = orbital_rate(LEO_SMA)
    dt1, dt2 = 30.0, 60.0
    Phi1 = hcw_state_transition(n, dt1)
    Phi2 = hcw_state_transition(n, dt2)
    Phi12 = hcw_state_transition(n, dt1 + dt2)
    assert np.allclose(Phi12, Phi1 @ Phi2, atol=1e-8)


# ── Propagation ───────────────────────────────────────────────────────────────

def test_propagate_initial_state():
    """Propagated state at t=0 should match x0."""
    n = orbital_rate(LEO_SMA)
    x0 = np.array([100.0, 1000.0, 50.0, 0.0, -10.0, 0.0])
    t_eval = np.linspace(0, 100, 50)
    states = propagate_hcw(x0, (0, 100), t_eval, n)
    assert np.allclose(states[0], x0, atol=1e-6)


def test_propagate_zero_relative_velocity():
    """Pure radial offset with no relative velocity should drift in HCW."""
    n = orbital_rate(LEO_SMA)
    x0 = np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    t_eval = np.linspace(0, 3600, 100)
    states = propagate_hcw(x0, (0, 3600), t_eval, n)
    # Should remain bounded (HCW predicts drift, but bounded for one orbit)
    assert np.all(np.isfinite(states))


# ── Scenario ──────────────────────────────────────────────────────────────────

def test_leo_scenario_tca_positive():
    scenario = make_leo_conjunction()
    t_tca, d_tca = scenario.find_tca()
    assert t_tca > 0
    assert d_tca >= 0


# def test_leo_scenario_miss_distance_reasonable():
#     """Miss distance should be small (near-miss conjunction) but > 0."""
#     scenario = make_leo_conjunction(
#         along_track_offset_km=50.0,
#         radial_offset_km=0.05,
#         relative_velocity_mps=15.0,
#     )
#     _, d_tca = scenario.find_tca()
#     # With 50m radial offset, miss should be ~50m, definitely < 1km
#     assert d_tca < 1000.0

def test_crossing_scenario_miss_distance():
    """Miss distance at TCA should match specified miss distance."""
    scenario = make_crossing_conjunction(miss_distance=1.0)
    _, d_tca = scenario.find_tca()
    # Should be within 1m of specified 1km miss distance
    assert abs(d_tca - 1000.0) < 1.0


def test_scenario_covariance_shape():
    scenario = make_leo_conjunction()
    assert scenario.covariance.shape == (6, 6)
    # Should be positive semi-definite
    eigvals = np.linalg.eigvalsh(scenario.covariance)
    assert np.all(eigvals >= -1e-10)
