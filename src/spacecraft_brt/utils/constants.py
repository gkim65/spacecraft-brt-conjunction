"""
Physical and operational constants for the spacecraft-BRT-conjunction project.

All values are in SI units (m, m/s, s) unless otherwise noted. No physical
constant should be hardcoded anywhere else in the codebase — import it from
here so there is a single source of truth (see claude.md conventions).
"""

# Earth gravitational parameter, mu = G * M_earth (m^3 / s^2).
# Source: WGS-84 / EGM-derived standard value.
MU_EARTH: float = 3.986004418e14

# Mean volumetric radius of Earth (m).
# Source: IUGG mean radius R_1 = (2a + b) / 3 for WGS-84.
R_EARTH: float = 6.371e6

# Default combined hard-body radius (m): sum of the primary and secondary
# bounding-sphere radii. Representative LEO value per claude.md conventions.
R_HBR_DEFAULT: float = 10.0

# Operational probability-of-collision threshold for a maneuver decision.
# Source: NASA CARA operational threshold (claude.md conventions).
PC_THRESHOLD: float = 1e-4