"""
Hill-Clohessy-Wiltshire (HCW) relative orbital dynamics.

Models the linearized relative motion between a chaser spacecraft and
a target (debris) object in the RTN (Radial-Transverse-Normal) frame,
assuming a circular reference orbit.

State vector: x = [dr, dt, dn, drdot, dtdot, dndot]
    dr    — radial separation (m)
    dt    — transverse (along-track) separation (m)
    dn    — normal (cross-track) separation (m)
    drdot — radial relative velocity (m/s)
    dtdot — transverse relative velocity (m/s)
    dndot — normal relative velocity (m/s)

Reference:
    Clohessy, W. H., & Wiltshire, R. S. (1960). Terminal guidance system
    for satellite rendezvous. Journal of the Aerospace Sciences, 27(9).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from ..utils.constants import MU_EARTH


def orbital_rate(sma: float) -> float:
    """
    Mean orbital angular rate for a circular orbit.

    Args:
        sma: Semi-major axis in meters.

    Returns:
        Mean motion n in rad/s.
    """
    return np.sqrt(MU_EARTH / sma**3)


def hcw_matrices(n: float) -> tuple[np.ndarray, np.ndarray]:
    """
    HCW state-space matrices for linearized relative dynamics.

    Continuous-time system: dx/dt = A @ x + B @ u

    Args:
        n: Mean orbital angular rate (rad/s).

    Returns:
        A: 6x6 dynamics matrix.
        B: 6x3 control input matrix (thrust in RTN frame).
    """
    A = np.array([
        [0,    0, 0, 1,  0, 0],
        [0,    0, 0, 0,  1, 0],
        [0,    0, 0, 0,  0, 1],
        [3*n**2, 0, 0, 0, 2*n, 0],
        [0,    0, 0, -2*n, 0, 0],
        [0,    0, -n**2, 0, 0, 0],
    ])

    B = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    return A, B


def hcw_state_transition(n: float, dt: float) -> np.ndarray:
    """
    Analytical HCW state transition matrix Phi(dt) via matrix exponential.

    x(t + dt) = Phi(dt) @ x(t)  (uncontrolled, u=0)

    Args:
        n:  Mean orbital angular rate (rad/s).
        dt: Time step in seconds.

    Returns:
        Phi: 6x6 state transition matrix.
    """
    A, _ = hcw_matrices(n)
    return expm(A * dt)


def propagate_hcw(
    x0: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray,
    n: float,
    u: np.ndarray | None = None,
) -> np.ndarray:
    """
    Propagate relative state under HCW dynamics using numerical integration.

    Args:
        x0:     Initial relative state [dr, dt, dn, drdot, dtdot, dndot] (m, m/s).
        t_span: (t_start, t_end) in seconds.
        t_eval: Times at which to evaluate the solution (seconds).
        n:      Mean orbital angular rate (rad/s).
        u:      Control input [ur, ut, un] in m/s^2. Constant over interval.
                If None, assumes uncontrolled (u = 0).

    Returns:
        states: (len(t_eval), 6) array of relative states.
    """
    A, B = hcw_matrices(n)
    u_vec = np.zeros(3) if u is None else np.asarray(u)

    def dynamics(t, x):
        return A @ x + B @ u_vec

    sol = solve_ivp(
        dynamics,
        t_span,
        x0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )

    return sol.y.T  # shape (len(t_eval), 6)


def relative_position(states: np.ndarray) -> np.ndarray:
    """Extract relative position [dr, dt, dn] from state array."""
    return states[:, :3]


def relative_velocity(states: np.ndarray) -> np.ndarray:
    """Extract relative velocity [drdot, dtdot, dndot] from state array."""
    return states[:, 3:]


def separation_distance(states: np.ndarray) -> np.ndarray:
    """
    Euclidean separation distance at each timestep.

    Args:
        states: (N, 6) state array.

    Returns:
        distances: (N,) array of separation distances in meters.
    """
    return np.linalg.norm(relative_position(states), axis=1)


def time_of_closest_approach(
    states: np.ndarray,
    t_eval: np.ndarray,
) -> tuple[float, float]:
    """
    Find time and distance of closest approach from a propagated trajectory.

    Args:
        states: (N, 6) state array from propagate_hcw.
        t_eval: (N,) time array corresponding to states.

    Returns:
        t_tca:  Time of closest approach (seconds).
        d_tca:  Separation distance at TCA (meters).
    """
    distances = separation_distance(states)
    idx = np.argmin(distances)
    return float(t_eval[idx]), float(distances[idx])
