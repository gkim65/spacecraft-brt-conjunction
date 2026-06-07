"""
Uncontrolled backward-reachable tube (BRT) for HCW conjunction analysis.

This module computes the *uncontrolled* BRT of the collision sphere under the
linearized HCW relative dynamics. It is the geometric set over which the CDM
covariance is integrated to obtain the time-evolving probability of collision
Pc(t) (see probability/pc_brt.py) and the latest-safe-maneuver-time t*.

Core insight (why no grid / no Riccati is needed here)
------------------------------------------------------
With no control (B = 0), HCW is an autonomous linear flow  dx/dt = A x. The
state at the final time T, starting from x at time t, is *exactly*

    x_T = Phi(T - t) @ x ,     Phi(dt) = expm(A * dt)

with no integration error -- Phi is the closed-form state transition matrix. A
relative state x is in the BRT at time t iff its forward image x_T lands inside
the collision sphere, i.e. iff l(x_T) <= 0. We therefore *define* the BRT value
function as the failure-set signed distance of the forward-propagated state:

    V(x, t) = l( Phi(T - t) @ x )

Geometrically: the BRT at time t is the collision sphere pulled back through the
linear flow over the remaining time T - t. Because relative velocity steers
where a state flows, the BRT is velocity-dependent even though the target set
l(x) is position-only.

Sign convention (inherited verbatim from reachability/target_set.py)
--------------------------------------------------------------------
    V(x, t) <  0   ->  inside the BRT  (state reaches the sphere by T -- UNSAFE)
    V(x, t) == 0   ->  on the BRT boundary
    V(x, t) >  0   ->  outside the BRT (safe over [t, T])

At the final time t = T we have Phi(0) = I, so V(x, T) = l(x) exactly: the BRT
reduces to the target set, as it must.

Time-direction bookkeeping
--------------------------
We propagate *forward* from t to T, and T >= t, so the propagation step is
dt = T - t >= 0. Flipping that sign runs the dynamics backward and silently
inverts which states are flagged unsafe -- the tests guard against this.

Reference:
    Bansal et al. (2017), Hamilton-Jacobi Reachability: A Brief Overview.
"""

import numpy as np

from ..dynamics.hcw import hcw_state_transition
from ..utils.constants import R_HBR_DEFAULT
from .target_set import signed_distance


def value(
    x: np.ndarray,
    t: float,
    T: float,
    n: float,
    hard_body_r: float = R_HBR_DEFAULT,
) -> np.ndarray:
    """
    Uncontrolled BRT value function V(x, t) = l(Phi(T - t) @ x).

    A state is inside the BRT (V < 0) when, propagated forward under the
    uncontrolled HCW flow from time t to the final time T, it lands inside the
    collision sphere.

    Args:
        x:           Relative state(s) in the RTN frame. Either a single
                     6-vector [dr, dt, dn, drdot, dtdot, dndot] (m, m/s) or a
                     batch of shape (N, 6).
        t:           Current time (s). Must satisfy t <= T.
        T:           Final time of the encounter horizon (s).
        n:           Mean orbital angular rate (rad/s).
        hard_body_r: Combined hard-body radius R_hbr (m).

    Returns:
        V(x, t) in meters, with the target_set sign convention (< 0 inside the
        BRT, == 0 on the boundary, > 0 safe). Scalar (0-d array) for a single
        state, or shape (N,) for a batch.
    """
    x = np.asarray(x, dtype=float)
    phi = hcw_state_transition(n, T - t)          # forward flow over remaining time
    # Apply Phi to each state. For a batch (N,6): x_T[i] = Phi @ x[i] == x @ Phi.T.
    # For a single (6,) vector this reduces to the same matmul.
    x_T = x @ phi.T
    return signed_distance(x_T, hard_body_r)


def brt_boundary_points(
    t: float,
    T: float,
    n: float,
    hard_body_r: float = R_HBR_DEFAULT,
    n_samples: int = 200,
    velocity: np.ndarray | None = None,
) -> np.ndarray:
    """
    Sample the V(., t) = 0 level set (the BRT boundary) as full 6D states.

    The BRT boundary is the preimage of the collision-sphere surface under the
    forward flow Phi(T - t). We sample the sphere surface (positions at distance
    R_hbr) at the final time T -- assigning a chosen relative velocity -- and
    pull each full 6D state back through Phi(T - t)^{-1} = Phi(t - T). The result
    is the set of states at time t whose forward image lands exactly on the
    sphere, i.e. V(x, t) = 0.

    The full 6D state is returned (not just dr-dt) because the flow couples
    position and velocity: the pulled-back state has a different velocity than
    the one assigned at T, and only the full state satisfies V = 0. For plotting
    in the RTN plane, take points[:, :2] (dr, dt) of the returned array.

    Args:
        t:           Current time (s).
        T:           Final time of the encounter horizon (s).
        n:           Mean orbital angular rate (rad/s).
        hard_body_r: Combined hard-body radius R_hbr (m).
        n_samples:   Number of points sampled around the sphere surface.
        velocity:    Relative velocity [drdot, dtdot, dndot] (m/s) assigned to
                     the sphere-surface states at time T before pull-back.
                     Defaults to zero. Different velocities trace different
                     slices of the (generally 6D) boundary.

    Returns:
        points: (n_samples, 6) array of boundary states at time t. Each row
                satisfies value(row, t, T, n, hard_body_r) == 0. Project to
                points[:, :2] for an RTN (dr, dt) boundary curve.
    """
    vel = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float)

    # Points on the collision-sphere surface in the dr-dt plane at the final time.
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples)
    boundary_T = np.zeros((n_samples, 6))
    boundary_T[:, 0] = hard_body_r * np.cos(theta)   # dr
    boundary_T[:, 1] = hard_body_r * np.sin(theta)   # dt
    boundary_T[:, 3:] = vel                          # assigned relative velocity

    # Pull back through the inverse flow: Phi(T - t)^{-1} = Phi(t - T).
    phi_inv = hcw_state_transition(n, t - T)
    return boundary_T @ phi_inv.T


def value_avoid(
    x: np.ndarray,
    t: float,
    T: float,
    n: float,
    hard_body_r: float = R_HBR_DEFAULT,
) -> np.ndarray:
    """
    [SEAM -- Step 8, not implemented] Controlled avoid-BRT value function.

    The *controlled* reach-avoid tube is the set of states from which the
    spacecraft cannot avoid the collision sphere even at maximum control
    authority. It solves the matrix Riccati ODE

        -dM/dt = A^T M + M A - M B B^T M ,   M(T) = I

    backward over the horizon, and is the correct object for the min-DeltaV
    Control Barrier Function controller (control/cbf.py).

    This is intentionally a separate object from `value` above: Pc(t) integrates
    covariance over the *uncontrolled* BRT (a probability question), while the
    avoid-BRT is a *control-authority* question. Do NOT integrate covariance
    over this set -- see docs/contribution-framing.md.

    Deferred to Step 8.
    """
    raise NotImplementedError(
        "value_avoid (controlled Riccati avoid-BRT) is deferred to Step 8 "
        "(CBF controller). Use `value` for the uncontrolled Pc(t) BRT."
    )