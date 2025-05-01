"""This module defines a mass profile where the convergence follows 
and elliptical Sersic profile.  This makes use of the MGE profile.
"""

__author__ = "CKrawczyk"

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
    '''Radial convergence function for a Sersic profile

    Parameters
    ----------
    r : float
        radius
    mass_to_light_ratio : float
        mass-to-light ratio
    intensity : float
        intensity
    effective_radius : float
        effective radius
    sersic_index : float
        sersic index

    Returns
    -------
    float
        Radial convergence for a Sersic profile
    '''
    b = sersic_constant(sersic_index)
    r_ = (r / effective_radius)**(1.0 / sersic_index)
    return mass_to_light_ratio * intensity * jnp.exp(-b * (r_ - 1.0))


# Subclass MGE to initialize with the radial function


class SersicEllipseKappa(MGE):
    def __init__(self):
        super().__init__(
            sersic_fn,
            'effective_radius',
            n_gauss=20,
            n_terms=28,
            sigma_start_mult=1/100,
            sigma_end_mult=20
        )

    # "override" these methods just to change the docstring
    def function(self, x, y, **kwargs):
        '''
        Returns the lensing potential for a mass with an elliptical Sersic convergence

        Parameters
        ----------
        x : jax.numpy.array
            coordinate on the sky
        y : jax.numpy.array
            coordinate on the sky
        e1 : float, keyword
            eccentricity modulus, must be given as a keyword
        e2 : float, keyword
            eccentricity modulus, must be given as a keyword
        center_x : float, keyword
            center of the profile, must be given as a keyword
        center_y : float, keyword
            center of the profile, must be given as a keyword
        mass_to_light_ratio : float, keyword
            mass-to-light ratio, must be given as a keyword
        intensity : float, keyword
            intensity, must be given as a keyword
        effective_radius : float, keyword
            effective radius, must be given as a keyword
        sersic_index : float, keyword
            sersic index, must be given as a keyword

        Returns
        -------
        jax.numpy.array
            Returns the lensing potential for a mass with an elliptical Sersic convergence
        '''
        return super().function(x, y, **kwargs)
    
    def derivatives(self, x, y, e1, e2, center_x=0, center_y=0, **kwargs):
        '''Returns the deflection angles for a mass with an elliptical Sersic convergence

        Parameters
        ----------
        x : jax.numpy.array
            coordinate on the sky
        y : jax.numpy.array
            coordinate on the sky
        e1 : float, keyword
            eccentricity modulus, must be given as a keyword
        e2 : float, keyword
            eccentricity modulus, must be given as a keyword
        center_x : float, keyword
            center of the profile, must be given as a keyword
        center_y : float, keyword
            center of the profile, must be given as a keyword
        mass_to_light_ratio : float, keyword
            mass-to-light ratio, must be given as a keyword
        intensity : float, keyword
            intensity, must be given as a keyword
        effective_radius : float, keyword
            effective radius, must be given as a keyword
        sersic_index : float, keyword
            sersic index, must be given as a keyword

        Returns
        -------
        jax.numpy.array
            Returns the deflection angles for a mass with an elliptical Sersic convergence
        '''
        return super().derivatives(x, y, e1, e2, center_x, center_y, **kwargs)
    
    def hessian(self, x, y, e1, e2, center_x=0, center_y=0, **kwargs):
        '''Returns the hessian with respect to position for a mass with an elliptical Sersic convergence

        Parameters
        ----------
        x : jax.numpy.array
            coordinate on the sky
        y : jax.numpy.array
            coordinate on the sky
        e1 : float, keyword
            eccentricity modulus, must be given as a keyword
        e2 : float, keyword
            eccentricity modulus, must be given as a keyword
        center_x : float, keyword
            center of the profile, must be given as a keyword
        center_y : float, keyword
            center of the profile, must be given as a keyword
        mass_to_light_ratio : float, keyword
            mass-to-light ratio, must be given as a keyword
        intensity : float, keyword
            intensity, must be given as a keyword
        effective_radius : float, keyword
            effective radius, must be given as a keyword
        sersic_index : float, keyword
            sersic index, must be given as a keyword

        Returns
        -------
        jax.numpy.array
            Returns the hessian with respect to position for a mass with an elliptical Sersic convergence
        '''
        return super().hessian(x, y, e1, e2, center_x, center_y, **kwargs)
