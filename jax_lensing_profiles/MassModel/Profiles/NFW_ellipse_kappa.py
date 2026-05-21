"""This module defines a mass profile where the convergence follows 
and elliptical NFW profile.  This makes use of the MGE profile.
The conventions of Keeton (2002, https://arxiv.org/pdf/astro-ph/0102341)
are used.
"""

__author__ = "WolfgangEnzi", "CKrawczyk", "astroskylee"

import jax.numpy as jnp

from .MGE import MGE, N_TERMS


def NFW_3D_fn(r, R_s, kappa_s, **_):
    '''Radial scaled mass function for a NFW profile

    Parameters
    ----------
    r : float
        radius
    R_s : float
        scale radius
    kappa_s : float
        amplitude of the convergence.  This is related to the amplitude
        of the mass profile by rho_s = kappa_s * Critical_Surface_Density / R_s

    Returns
    -------
    float
        scaled mass value for a 3D NFW profile
    '''
    x = r / R_s
    return kappa_s / (r * (1 + x)**2)


# Subclass MGE to initialize with the radial function
class NFWEllipseKappa(MGE):
    def __init__(self):
        super().__init__(
            NFW_3D_fn,
            'R_s',
            n_gauss=20,
            n_terms=N_TERMS,
            sigma_start_mult=1/500,
            sigma_end_mult=20,
            three_d=True
        )

    # "override" these methods just to change the docstring
    def function(self, x, y, **kwargs):
        '''Returns the lensing potential for a mass with an elliptical NFW convergence

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
        R_s : float, keyword
            scale radius, must be given as a keyword
        kappa_s : float, keyword
            amplitude of the convergence, must be given as a keyword.
            This is related to the amplitude of the mass profile by
            rho_s = kappa_s * Critical_Surface_Density / R_s

        Returns
        -------
        jax.numpy.array
            Returns the lensing potential for a mass with an elliptical NFW convergence
        '''
        return super().function(x, y, **kwargs)

    def derivatives(self, x, y, e1, e2, center_x=0, center_y=0, **kwargs):
        '''Returns the deflection angles for a mass with an elliptical NFW convergence

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
        R_s : float, keyword
            scale radius, must be given as a keyword
        kappa_s : float, keyword
            amplitude of the convergence, must be given as a keyword.
            This is related to the amplitude of the mass profile by
            rho_s = kappa_s * Critical_Surface_Density / R_s

        Returns
        -------
        jax.numpy.array
            Returns the deflection angles for a mass with an elliptical NFW convergence
        '''
        return super().derivatives(x, y, e1, e2, center_x, center_y, **kwargs)

    def hessian(self, x, y, e1, e2, center_x=0, center_y=0, **kwargs):
        '''Returns the hessian with respect to position for a mass with an elliptical NFW convergence

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
        R_s : float, keyword
            scale radius, must be given as a keyword
        kappa_s : float, keyword
            amplitude of the convergence, must be given as a keyword.
            This is related to the amplitude of the mass profile by
            rho_s = kappa_s * Critical_Surface_Density / R_s

        Returns
        -------
        jax.numpy.array
            Returns the hessian with respect to position for a mass with an elliptical NFW convergence
        '''
        return super().hessian(x, y, e1, e2, center_x, center_y, **kwargs)
