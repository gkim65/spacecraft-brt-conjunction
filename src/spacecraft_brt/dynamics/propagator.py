"""
Full two-body propagator using brahe for HCW validation.

Propagates both spacecraft and debris in ECI using brahe's numerical
propagator, then computes relative state in RTN frame. Compared against
HCW to validate the linear approximation for a given conjunction geometry.

Usage:
    from spacecraft_brt.dynamics.propagator import BrahePropagator
    prop = BrahePropagator(sma=6.921e6)
    t, rel_states = prop.propagate_relative(x0_eci_sc, x0_eci_debris, t_span)
"""

import numpy as np

try:
    import brahe
    from brahe import Epoch, R_EARTH
    from brahe.orbit_propagation import propagate as brahe_propagate
    BRAHE_AVAILABLE = True
except ImportError:
    BRAHE_AVAILABLE = False


def eci_to_rtn(r_ref: np.ndarray, v_ref: np.ndarray) -> np.ndarray:
    """
    Compute the ECI-to-RTN rotation matrix.

    RTN frame:
        R — radial (along position vector)
        T — transverse (along-track, in orbit plane)
        N — normal (cross-track, orbit normal)

    Args:
        r_ref: ECI position of reference orbit (m), shape (3,).
        v_ref: ECI velocity of reference orbit (m/s), shape (3,).

    Returns:
        R_eci_to_rtn: 3x3 rotation matrix.
    """
    r_hat = r_ref / np.linalg.norm(r_ref)
    h = np.cross(r_ref, v_ref)
    n_hat = h / np.linalg.norm(h)
    t_hat = np.cross(n_hat, r_hat)

    return np.array([r_hat, t_hat, n_hat])  # rows are RTN unit vectors


def absolute_to_relative_rtn(
    r_sc: np.ndarray,
    v_sc: np.ndarray,
    r_debris: np.ndarray,
    v_debris: np.ndarray,
) -> np.ndarray:
    """
    Convert absolute ECI states to relative RTN state vector.

    Args:
        r_sc, v_sc:         Spacecraft ECI position/velocity (m, m/s).
        r_debris, v_debris: Debris ECI position/velocity (m, m/s).

    Returns:
        x_rel: Relative state [dr, dt, dn, drdot, dtdot, dndot] in RTN (m, m/s).
    """
    R = eci_to_rtn(r_sc, v_sc)

    dr_eci = r_debris - r_sc
    dv_eci = v_debris - v_sc

    # Rotate into RTN
    dr_rtn = R @ dr_eci
    dv_rtn = R @ dv_eci

    return np.concatenate([dr_rtn, dv_rtn])


class BrahePropagator:
    """
    Validates HCW relative motion against full two-body propagation via brahe.

    This is intentionally a validation tool, not the primary dynamics model.
    HCW is used for BRT computation; brahe propagation checks the approximation
    error for a given scenario geometry and timescale.
    """

    def __init__(self):
        if not BRAHE_AVAILABLE:
            raise ImportError(
                "brahe is required for full two-body validation. "
                "Install with: uv add brahe"
            )

    def propagate_relative(
        self,
        r0_sc: np.ndarray,
        v0_sc: np.ndarray,
        r0_debris: np.ndarray,
        v0_debris: np.ndarray,
        t_eval: np.ndarray,
    ) -> np.ndarray:
        """
        Propagate both objects under two-body dynamics and return relative state.

        Args:
            r0_sc:     Initial spacecraft ECI position (m).
            v0_sc:     Initial spacecraft ECI velocity (m/s).
            r0_debris: Initial debris ECI position (m).
            v0_debris: Initial debris ECI velocity (m/s).
            t_eval:    Times at which to evaluate (seconds from epoch).

        Returns:
            rel_states: (len(t_eval), 6) relative state array in RTN (m, m/s).
        """
        # NOTE: brahe propagation API to be confirmed against installed version.
        # This is a placeholder structure — update once brahe propagator
        # API is confirmed from docs/examples.
        raise NotImplementedError(
            "Brahe propagator integration pending API confirmation. "
            "See notebooks/01_brahe_validation.ipynb for setup."
        )

    def hcw_error(
        self,
        hcw_states: np.ndarray,
        brahe_states: np.ndarray,
    ) -> dict:
        """
        Compute HCW approximation error relative to full two-body propagation.

        Args:
            hcw_states:   (N, 6) HCW propagated states.
            brahe_states: (N, 6) brahe propagated states.

        Returns:
            Dictionary with position and velocity error statistics (m, m/s).
        """
        pos_error = np.linalg.norm(
            hcw_states[:, :3] - brahe_states[:, :3], axis=1
        )
        vel_error = np.linalg.norm(
            hcw_states[:, 3:] - brahe_states[:, 3:], axis=1
        )

        return {
            "pos_error_max_m":  float(np.max(pos_error)),
            "pos_error_rms_m":  float(np.sqrt(np.mean(pos_error**2))),
            "vel_error_max_mps": float(np.max(vel_error)),
            "vel_error_rms_mps": float(np.sqrt(np.mean(vel_error**2))),
        }
