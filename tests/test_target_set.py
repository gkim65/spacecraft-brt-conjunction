"""Tests for the collision-sphere target set, l(x)."""

import numpy as np
import pytest

from spacecraft_brt.reachability.target_set import signed_distance
from spacecraft_brt.utils.constants import R_HBR_DEFAULT


def test_origin_is_minus_hbr():
    """At the origin, separation is 0 so l = -R_hbr (deepest inside)."""
    x = np.zeros(6)
    assert signed_distance(x) == pytest.approx(-R_HBR_DEFAULT)


def test_on_sphere_surface_is_zero():
    """A point exactly R_hbr away (in any direction) lies on the boundary."""
    x = np.zeros(6)
    x[1] = R_HBR_DEFAULT  # along-track, magnitude R_hbr
    assert signed_distance(x) == pytest.approx(0.0)


def test_far_point_is_positive():
    """A well-separated state is safe -> l > 0, equal to range minus R_hbr."""
    x = np.array([300.0, 400.0, 0.0, 0.0, 0.0, 0.0])  # 500 m range
    assert signed_distance(x) == pytest.approx(500.0 - R_HBR_DEFAULT)


def test_velocity_is_ignored():
    """l(x) depends only on position; velocity must not change it."""
    x_slow = np.array([0.0, R_HBR_DEFAULT, 0.0, 0.0, 0.0, 0.0])
    x_fast = np.array([0.0, R_HBR_DEFAULT, 0.0, 7000.0, -3000.0, 50.0])
    assert signed_distance(x_slow) == pytest.approx(signed_distance(x_fast))


def test_custom_hard_body_radius():
    """The hard-body radius is configurable."""
    x = np.array([0.0, 25.0, 0.0, 0.0, 0.0, 0.0])
    assert signed_distance(x, hard_body_r=25.0) == pytest.approx(0.0)


def test_batched_input_shape_and_values():
    """A batch of states returns shape (N,) with per-row signed distances."""
    batch = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],            # l = -R_hbr
        [0.0, R_HBR_DEFAULT, 0.0, 0.0, 0.0, 0.0],   # l = 0
        [300.0, 400.0, 0.0, 0.0, 0.0, 0.0],         # l = 500 - R_hbr
    ])
    out = signed_distance(batch)
    assert out.shape == (3,)
    np.testing.assert_allclose(
        out, [-R_HBR_DEFAULT, 0.0, 500.0 - R_HBR_DEFAULT]
    )