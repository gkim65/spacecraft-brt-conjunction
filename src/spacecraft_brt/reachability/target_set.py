"""
Target (failure) set for conjunction reachability.

The failure set is the collision sphere: a primary-debris collision occurs
when their relative separation drops below the combined hard-body radius
R_hbr. Collision is a purely *geometric* condition, so the set is defined on
the relative position [dr, dt, dn] only and is independent of relative velocity.

We represent the set by an implicit signed-distance function l(x):

    l(x) = || [dr, dt, dn] ||  -  R_hbr

with the convention

    l(x) <  0   ->  inside the collision sphere   (FAILURE)
    l(x) == 0   ->  on the sphere surface         (boundary)
    l(x) >  0   ->  safe separation

The failure set is then L = { x : l(x) <= 0 }. The backward-reachable-tube
value function V(x, t) (see reachability/hopf.py) inherits this same sign
convention -- "<= 0 means unsafe" -- so the target set and the BRT compose
cleanly: at the final time the BRT reduces to exactly this set.
"""

import numpy as np

from ..utils.constants import R_HBR_DEFAULT

# Indices of the position components within the 6D RTN state
# x = [dr, dt, dn, drdot, dtdot, dndot].
_POSITION_SLICE = slice(0, 3)


def signed_distance(x: np.ndarray, hard_body_r: float = R_HBR_DEFAULT) -> np.ndarray:
    """
    Signed distance to the collision-sphere surface.

    Computes l(x) = ||[dr, dt, dn]|| - R_hbr from the relative position only;
    relative velocity is ignored because collision is a geometric condition.

    Args:
        x:           Relative state(s) in the RTN frame. Either a single
                     6-vector [dr, dt, dn, drdot, dtdot, dndot] (m, m/s) or a
                     batch of shape (N, 6).
        hard_body_r: Combined hard-body radius R_hbr (m). Defaults to the
                     project standard R_HBR_DEFAULT.

    Returns:
        Signed distance l(x) in meters. A scalar (0-d array) for a single
        state, or shape (N,) for a batch. Negative inside the collision
        sphere, zero on its surface, positive in the safe region.
    """
    x = np.asarray(x, dtype=float)
    pos = x[..., _POSITION_SLICE]
    separation = np.linalg.norm(pos, axis=-1)
    return separation - hard_body_r