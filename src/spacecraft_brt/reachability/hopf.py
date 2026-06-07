"""
Uncontrolled backward-reachable TUBE (BRT) for HCW conjunction analysis.

This module computes the *uncontrolled* BRT of the collision sphere under the
linearized HCW relative dynamics. It is the geometric set over which the CDM
covariance is integrated to obtain the time-evolving probability of collision
Pc(t) (see probability/pc_brt.py) and the latest-safe-maneuver-time t*.

Tube vs. set (the central distinction -- see docs/brt-tube-upgrade-spec.md)
--------------------------------------------------------------------------
With no control (B = 0), HCW is an autonomous linear flow  dx/dt = A x. The
state at absolute time tau, starting from x at time t, is *exactly*

    x(tau) = Phi(tau - t) @ x ,     Phi(s) = expm(A * s)

with no integration error -- Phi is the closed-form state transition matrix.

A state x at time t should be flagged UNSAFE if its trajectory enters the
collision sphere at ANY point in the remaining window [t, T] -- not only at the
final instant T. "Reach the target at some tau in [t, T]" is the backward
reachable *tube*; "be in the target exactly at T" is the backward reachable
*set at a single time*. The min-over-time value function encodes the tube:

    V(x, t) = min over tau in [t, T] of  l( Phi(tau - t) @ x )

If the trajectory dips inside the sphere at any tau, that term is < 0 and the
min is < 0. This is REQUIRED for our crossing scenario: at ~7 km/s the objects
are inside a 10 m sphere for only ~milliseconds and essentially never exactly
at T, so the single-time check l(Phi(T-t) @ x) almost always misses the
collision. (The previous version of this module computed only that single-time
set; this module computes the tube.)

The full-window [0, T] answer is the special case t = 0; no separate
computation is needed. Note V(x, t) and V(x0, 0) describe the same trajectory
but are NOT equal: x is the state AT time t and its window is [t, T] only --
the already-elapsed [0, t] portion is deliberately excluded ("remaining-window"
risk). Hence we always propagate from the state x given at time t, by
Phi(tau - t), never from a t = 0 state.

Sign convention (inherited verbatim from reachability/target_set.py)
--------------------------------------------------------------------
    V(x, t) <  0   ->  inside the BRT  (trajectory reaches the sphere -- UNSAFE)
    V(x, t) == 0   ->  on the BRT boundary
    V(x, t) >  0   ->  outside the BRT (safe over the whole window [t, T])

At the final time t = T the window is the single point {T}, Phi(0) = I, so
V(x, T) = l(x) exactly: the BRT reduces to the target set, as it must.

Monotonicity
------------
As t increases the min is taken over a SHORTER window, so along a trajectory
V(x(t), t) is non-decreasing in t (already-passed collision opportunities drop
off; remaining risk only shrinks). This is what makes the latest-safe-maneuver
time t* well defined. Guarded by a unit test.

Time resolution -- the real correctness knob
--------------------------------------------
The min is over CONTINUOUS tau; we approximate it by sampling. At 7 km/s
through a 10 m sphere the in-sphere dwell is ~milliseconds, so a coarse uniform
tau grid can STEP OVER the passage and silently report SAFE (undercounting Pc).
The fix is two-stage and physics-sized:

  1. COARSE BRACKET. ||pos(tau)||^2 is a smooth function of tau (sums of
     1, s, sin(ns), cos(ns) for HCW). Its valley near a close approach has a
     time width set by the relative speed, NOT by the ms in-sphere dwell:
     the trajectory takes ~ L_basin / v_rel to traverse a sub-km neighbourhood
     of the origin. We choose the coarse spacing ds < L_basin / v_rel so a
     valley can never fall entirely between two samples. ds (hence n_time)
     therefore ADAPTS to (T - t) and to the relative speed of the batch, rather
     than being a fixed count -- a long Pc(t) window gets proportionally more
     samples at the SAME time resolution.

  2. REFINE. The HCW flow is oscillatory, so over a multi-orbit horizon there
     are MANY close-approach valleys (one per orbit), not one. We locate EVERY
     interior local minimum of the coarse ||pos||^2 curve and Brent-refine each
     to its exact closest approach, then take the global minimum. Refining only
     the single coarse argmin would find one valley and could miss the deepest.

The coarse grid is shared across all samples in a batch (the Phi(tau - t)
matrices do not depend on the sample), so the cost is K matmuls + an (N, K)
reduction, not N*K solves -- see Step 5 (Pc Monte Carlo).

Reference:
    Bansal et al. (2017), Hamilton-Jacobi Reachability: A Brief Overview.
"""

import numpy as np
from scipy.optimize import minimize_scalar

from ..dynamics.hcw import hcw_state_transition
from ..utils.constants import R_HBR_DEFAULT
from .target_set import signed_distance

# Spatial scale of the close-approach "valley" used to size the coarse tau grid.
# The bracket spacing must be finer than (this scale) / v_rel so a valley can
# never fall entirely between two coarse samples (step-over guard). A few
# hundred metres is comfortably sub-km and many times R_hbr, so the valley in
# ||pos(tau)|| at this scale is wide in tau even when the in-sphere dwell is ms.
_VALLEY_SCALE_M: float = 300.0

# Floor / cap on the coarse-grid sample count, to keep a single value() call
# bounded regardless of window length or relative speed.
_N_TIME_MIN: int = 64
_N_TIME_MAX: int = 200_000

# Relative-speed floor (m/s) when sizing the grid, so a near-static batch does
# not demand an absurdly coarse (or, via the cap, meaningless) grid.
_V_REL_FLOOR_MPS: float = 1.0


def _coarse_n_time(x: np.ndarray, window: float) -> int:
    """
    Number of coarse tau samples to bracket every close-approach valley.

    Sizes the uniform tau grid so its spacing ds = window / (n_time - 1) is
    finer than the valley traversal time L_basin / v_rel, using the FASTEST
    sample in the batch (conservative -- guarantees every sample is bracketed).

    Args:
        x:      Relative state(s), shape (6,) or (N, 6) (m, m/s).
        window: Remaining-window length T - t (s), assumed >= 0.

    Returns:
        Coarse sample count, clamped to [_N_TIME_MIN, _N_TIME_MAX].
    """
    if window <= 0.0:
        return 1
    vel = np.atleast_2d(x)[:, 3:]
    v_rel = float(np.max(np.linalg.norm(vel, axis=-1)))
    v_rel = max(v_rel, _V_REL_FLOOR_MPS)
    ds = _VALLEY_SCALE_M / v_rel                      # required spacing (s)
    n = int(np.ceil(window / ds)) + 1
    return int(np.clip(n, _N_TIME_MIN, _N_TIME_MAX))


def _propagated_positions(x2d: np.ndarray, phis: np.ndarray) -> np.ndarray:
    """
    Propagate a batch of states through a shared set of transition matrices.

    Args:
        x2d:  (N, 6) batch of states at time t.
        phis: (K, 6, 6) transition matrices Phi(tau_k - t).

    Returns:
        positions: (N, K, 3) propagated relative positions [dr, dt, dn].
    """
    # x(tau_k)[i] = Phi_k @ x2d[i]; batch over both i and k via einsum.
    states = np.einsum("kab,ib->ika", phis, x2d)     # (N, K, 6)
    return states[..., :3]


def value(
    x: np.ndarray,
    t: float,
    T: float,
    n: float,
    hard_body_r: float = R_HBR_DEFAULT,
    n_time: int | None = None,
    method: str = "refine",
) -> np.ndarray:
    """
    Uncontrolled BRT (tube) value V(x, t) = min_{tau in [t,T]} l(Phi(tau-t) @ x).

    A state is inside the BRT (V < 0) when its uncontrolled HCW trajectory
    enters the collision sphere at ANY time tau in the remaining window [t, T]
    -- not only at the final time T. See the module docstring for the tube-vs-
    set distinction and the time-resolution strategy.

    Args:
        x:           Relative state(s) in the RTN frame. Either a single
                     6-vector [dr, dt, dn, drdot, dtdot, dndot] (m, m/s) or a
                     batch of shape (N, 6).
        t:           Current time (s). Must satisfy t <= T.
        T:           Final time of the encounter horizon (s).
        n:           Mean orbital angular rate (rad/s).
        hard_body_r: Combined hard-body radius R_hbr (m).
        n_time:      Override for the coarse tau-grid sample count. If None
                     (default), it is sized from the window length and the
                     batch's relative speed (see _coarse_n_time).
        method:      'refine' (default) -- coarse bracket then Brent-refine every
                     local minimum to the exact closest approach (robust against
                     step-over and multi-orbit valleys). 'grid' -- coarse grid
                     minimum only (faster, for diagnostics; can step over).

    Returns:
        V(x, t) in meters, with the target_set sign convention (< 0 inside the
        BRT, == 0 on the boundary, > 0 safe). Scalar (0-d array) for a single
        state, or shape (N,) for a batch.
    """
    x = np.asarray(x, dtype=float)
    single = x.ndim == 1
    x2d = np.atleast_2d(x)                            # (N, 6)
    window = T - t

    # Degenerate window {T}: V reduces to l(x) exactly (Phi(0) = I).
    if window <= 0.0:
        out = signed_distance(x2d, hard_body_r)
        return out[0] if single else out

    # Stage 1 -- coarse bracket. Shared tau grid across the whole batch.
    n_coarse = _coarse_n_time(x2d, window) if n_time is None else int(n_time)
    s_grid = np.linspace(0.0, window, n_coarse)       # tau - t in [0, window]
    phis = np.stack([hcw_state_transition(n, s) for s in s_grid])  # (K,6,6)
    pos = _propagated_positions(x2d, phis)            # (N, K, 3)
    dist2 = np.einsum("ika,ika->ik", pos, pos)        # (N, K) = ||pos||^2

    if method == "grid":
        min_dist = np.sqrt(np.min(dist2, axis=1))
        out = min_dist - hard_body_r
        return out[0] if single else out
    if method != "refine":
        raise ValueError(f"method must be 'refine' or 'grid', got {method!r}")

    # Stage 2 -- refine. For each sample, Brent-refine every interior local
    # minimum of the coarse ||pos||^2 curve and take the global minimum.
    out = np.empty(x2d.shape[0])
    for i in range(x2d.shape[0]):
        di = dist2[i]
        best = di.min()
        # Interior local minima: strictly lower than both neighbours. Each is
        # bracketed by (s_grid[k-1], s_grid[k+1]) -- a valid Brent bracket.
        interior = np.flatnonzero(
            (di[1:-1] <= di[:-2]) & (di[1:-1] <= di[2:])
        ) + 1
        # Always also refine around the global coarse argmin (covers the case
        # where the deepest valley sits at, or adjacent to, a window endpoint).
        cands = set(interior.tolist())
        cands.add(int(np.argmin(di)))
        for k in cands:
            lo = s_grid[max(k - 1, 0)]
            hi = s_grid[min(k + 1, n_coarse - 1)]
            if hi <= lo:
                continue
            res = minimize_scalar(
                lambda s, xi=x2d[i]: float(
                    np.sum((hcw_state_transition(n, s) @ xi)[:3] ** 2)
                ),
                bounds=(lo, hi),
                method="bounded",
                options={"xatol": 1e-6},
            )
            best = min(best, res.fun)
        out[i] = np.sqrt(max(best, 0.0)) - hard_body_r

    return out[0] if single else out


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