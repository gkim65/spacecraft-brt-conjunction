"""
Conjunction scenario definition and setup.

Defines a standard two-body conjunction scenario between an active
spacecraft (primary) and a non-maneuverable debris object (secondary).
Initial conditions are specified in the RTN relative frame.
"""

from dataclasses import dataclass, field
import numpy as np
from .hcw import orbital_rate, propagate_hcw, time_of_closest_approach, hcw_state_transition


@dataclass
class ConjunctionScenario:
    """
    Defines a spacecraft-debris conjunction scenario.

    Attributes:
        sma:          Semi-major axis of reference orbit (m).
        x0:           Initial relative state [dr, dt, dn, drdot, dtdot, dndot].
        t_horizon:    Time horizon to propagate (seconds). Should cover TCA.
        hard_body_r:  Combined hard-body radius for collision (m). Typically
                      sum of spacecraft and debris bounding sphere radii.
        covariance:   6x6 combined covariance matrix in RTN frame (m^2, m^2/s^2).
                      Rows/cols ordered as [dr, dt, dn, drdot, dtdot, dndot].
        name:         Optional scenario label.
    """
    sma: float
    x0: np.ndarray
    t_horizon: float
    hard_body_r: float = 10.0  # 10m combined hard-body radius (typical LEO)
    covariance: np.ndarray = field(default_factory=lambda: np.eye(6))
    name: str = "default"

    def __post_init__(self):
        self.x0 = np.asarray(self.x0, dtype=float)
        self.covariance = np.asarray(self.covariance, dtype=float)
        assert self.x0.shape == (6,), "x0 must be a 6-element state vector"
        assert self.covariance.shape == (6, 6), "covariance must be 6x6"

    @property
    def n(self) -> float:
        """Mean orbital angular rate (rad/s)."""
        return orbital_rate(self.sma)

    def propagate(self, n_steps: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """
        Propagate the uncontrolled relative trajectory over t_horizon.

        Args:
            n_steps: Number of time steps.

        Returns:
            t_eval:  (n_steps,) time array (seconds).
            states:  (n_steps, 6) relative state array.
        """
        t_eval = np.linspace(0, self.t_horizon, n_steps)
        states = propagate_hcw(self.x0, (0, self.t_horizon), t_eval, self.n)
        return t_eval, states
    

    def find_tca(self, n_steps: int = 10000) -> tuple[float, float]:
        """
        Find time and distance of closest approach.

        Returns:
            t_tca: Time of closest approach (seconds).
            d_tca: Miss distance at TCA (meters).
        """
        t_eval, states = self.propagate(n_steps)
        return time_of_closest_approach(states, t_eval)

    def summary(self) -> str:
        t_tca, d_tca = self.find_tca()
        return (
            f"Scenario: {self.name}\n"
            f"  SMA:          {self.sma/1e3:.1f} km\n"
            f"  Mean motion:  {self.n*1e3:.4f} mrad/s\n"
            f"  T horizon:    {self.t_horizon/3600:.2f} hr\n"
            f"  TCA:          {t_tca/3600:.3f} hr ({t_tca:.1f} s)\n"
            f"  Miss distance:{d_tca:.1f} m\n"
            f"  Hard-body R:  {self.hard_body_r:.1f} m\n"
            f"  x0:           {self.x0}\n"
        )


def make_leo_conjunction(
    altitude_km: float = 550.0,
    along_track_offset_km: float = 50.0,
    radial_offset_km: float = 0.1,
    relative_velocity_mps: float = 15.0,
    hard_body_r: float = 10.0,
    covariance_scale: float = 1.0,
) -> ConjunctionScenario:
    """
    Build a representative LEO head-on conjunction scenario.

    The debris approaches from along-track with a small radial offset,
    producing a close approach near TCA.

    Args:
        altitude_km:           Orbit altitude in km.
        along_track_offset_km: Initial along-track separation (km).
        radial_offset_km:      Initial radial offset (km).
        relative_velocity_mps: Relative along-track closing speed (m/s).
        hard_body_r:           Combined hard-body radius (m).
        covariance_scale:      Scale factor for default covariance.

    Returns:
        ConjunctionScenario configured for the specified encounter.
    """
    R_EARTH = 6.371e6  # m
    sma = R_EARTH + altitude_km * 1e3

    x0 = np.array([
        radial_offset_km * 1e3,       # dr  — small radial offset
        along_track_offset_km * 1e3,  # dt  — along-track separation
        0.0,                           # dn  — no cross-track offset
        0.0,                           # drdot
        -relative_velocity_mps,        # dtdot — closing velocity
        0.0,                           # dndot
    ])

    # Time horizon: long enough that TCA occurs within it
    t_horizon = (along_track_offset_km * 1e3 / relative_velocity_mps) * 1.5

    # Representative CDM covariance — anisotropic, along-track dominated
    # Units: m^2 for position, (m/s)^2 for velocity
    sigma_r  = 50.0    * covariance_scale   # m
    sigma_t  = 500.0   * covariance_scale   # m  (along-track much larger)
    sigma_n  = 100.0   * covariance_scale   # m
    sigma_rd = 0.05    * covariance_scale   # m/s
    sigma_td = 0.1     * covariance_scale   # m/s
    sigma_nd = 0.05    * covariance_scale   # m/s

    cov = np.diag([
        sigma_r**2, sigma_t**2, sigma_n**2,
        sigma_rd**2, sigma_td**2, sigma_nd**2,
    ])

    return ConjunctionScenario(
        sma=sma,
        x0=x0,
        t_horizon=t_horizon,
        hard_body_r=hard_body_r,
        covariance=cov,
        name=f"LEO {altitude_km:.0f}km head-on",
    )


def make_crossing_conjunction(
    altitude_km: float = 550.0,
    relative_velocity_mps: float = 7000.0,
    hard_body_r: float = 10.0,
    covariance_scale: float = 1.0,
    miss_distance: float = 1.0,
    orbital_periods: float = 15.0,
) -> ConjunctionScenario:
    """
    Build a representative LEO crossing conjunction scenario.

    The debris approaches from along-track with a small radial offset,
    producing a close approach near TCA.

    Args:
        altitude_km:           Orbit altitude in km.
        relative_velocity_mps: Relative closing speed (m/s).
        hard_body_r:           Combined hard-body radius (m).
        covariance_scale:      Scale factor for default covariance.
        miss_distance:         Miss distance in km

    Returns:
        ConjunctionScenario configured for the specified encounter.
    """
    R_EARTH = 6.371e6  # m
    sma = R_EARTH + altitude_km * 1e3
    dr = miss_distance * 1e3  # convert km to m

    # At TCA for a crossing conjunction
    x_tca = [
        dr,   # dr  — small radial offset (the miss distance)
        0.0,             # dt  — at the crossing point, along-track = 0
        0.0,             # dn  — cross-track = 0 (or small)
        0.0,             # drdot — no radial closing speed at TCA
        0.0,             # dtdot — no along-track relative velocity
        relative_velocity_mps,          # dndot — relative velocity mostly cross-track (crossing!)
    ]

    # Time horizon: long enough that TCA occurs within it
    t_horizon = orbital_periods * (2 * np.pi / orbital_rate(sma))  # orbital periods >> around 24 hours
    
    # Representative CDM covariance — anisotropic, along-track dominated
    # Units: m^2 for position, (m/s)^2 for velocity
    sigma_r  = 50.0    * covariance_scale   # m
    sigma_t  = 500.0   * covariance_scale   # m  (along-track much larger)
    sigma_n  = 100.0   * covariance_scale   # m
    sigma_rd = 0.05    * covariance_scale   # m/s
    sigma_td = 0.1     * covariance_scale   # m/s
    sigma_nd = 0.05    * covariance_scale   # m/s

    cov = np.diag([
        sigma_r**2, sigma_t**2, sigma_n**2,
        sigma_rd**2, sigma_td**2, sigma_nd**2,
    ])
    Phi = hcw_state_transition(orbital_rate(sma), -t_horizon)
    x0 = Phi @ np.array(x_tca)
    return ConjunctionScenario(
        sma=sma,
        x0=x0,
        t_horizon=t_horizon,
        hard_body_r=hard_body_r,
        covariance=cov,
        name=f"LEO {altitude_km:.0f}km crossing-conjunction",
    )
