"""This module defines a mass profile where the convergence follows 
and elliptical NFW profile.  This makes use of the MGE profile.
"""

import jax.numpy as jnp

from .MGE_jax import MGE
from jax_lensing_profiles.Utility.f_function_jax import J


def NFW_fn(r, Rs, kappa_s, **_):
    x = r / Rs
    return 2 * kappa_s * J(x)


class NFWEllipseKappa(MGE):
    def __init__(self):
        super().__init__(
            NFW_fn,
            'Rs',
            n_gauss=20,
            n_terms=28,
            sigma_start_mult=1/500,
            sigma_end_mult=20
        )
