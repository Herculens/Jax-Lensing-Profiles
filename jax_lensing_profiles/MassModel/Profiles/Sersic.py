"""This module defines a mass profile where the convergence follows 
and elliptical Sersic profile.  This makes use of the MGE profile.
"""

import jax.numpy as jnp

from .MGE_jax import MGE


def sersic_constant(sersic_index):
    return (
        (2 * sersic_index)
        - (1.0 / 3.0)
        + (4.0 / (405.0 * sersic_index))
        + (46.0 / (25515.0 * sersic_index**2))
        + (131.0 / (1148175.0 * sersic_index**3))
        - (2194697.0 / (30690717750.0 * sersic_index**4))
    )


def sersic_fn(r, mass_to_light_ratio, intensity, effective_radius, sersic_index, **_):
    b = sersic_constant(sersic_index)
    r_ = (r / effective_radius)**(1.0 / sersic_index)
    return mass_to_light_ratio * intensity * jnp.exp(-b * (r_ - 1.0))


SersicEllipseKappa = MGE(
    sersic_fn,
    'effective_radius',
    n_gauss=20,
    n_terms=28,
    sigma_start_mult=1/100,
    sigma_end_mult=20
)
