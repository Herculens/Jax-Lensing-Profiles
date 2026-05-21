"""This module defines a mass profile where the convergence follows 
and dPIE profile.  This makes use of the MGE profile.
The 3D density profile is taken from Eliasdottir (2018,
https://arxiv.org/pdf/0710.5636).
"""

__author__ = "WolfgangEnzi", "CKrawczyk", "astroskylee"

import jax.numpy as jnp

from .MGE import MGE, N_TERMS


def dPIE_3D_fn(r, R_s, R_a, rho_0, **_):
    '''Radial scaled mass function for a dPIE profile

    Parameters
    ----------
    r : float
        radius
    R_s : float
        scale radius
    R_a : float
        core radius
    kappa_0 : float
        amplitude of the convergence.  This is related to the amplitude
        of the mass profile by rho_0 = kappa_0 * Critical_Surface_Density * (R_a + R_s) / (pi * R_a * R_s)

    Returns
    -------
    float
        scaled mass value for a dPIE profile
    '''
    x1 = r / R_s
    x2 = r / R_a
    # rho_0 = kappa_0 * (R_a + R_s) / (jnp.pi * R_a * R_s)
    return rho_0 / ((1 + x1**2) * (1 + x2**2))


# Subclass MGE to initialize with the radial function
class dPIEEllipseKappa(MGE):
    def __init__(self):
        super().__init__(
            dPIE_3D_fn,
            'R_s',
            n_gauss=20,
            n_terms=N_TERMS,
            sigma_start_mult=1/500,
            sigma_end_mult=20,
            three_d=True
        )

    # "override" these methods just to change the docstring
    def function(self, x, y, **kwargs):
        '''Returns the lensing potential for a mass with an elliptical dPIE
        convergence

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
        R_a : float
            core radius
        kappa_0 : float, keyword
            amplitude of the convergence, must be given as a keyword.
            This is related to the amplitude of the mass profile by
            rho_0 = kappa_0 * Critical_Surface_Density * (R_a + R_s) / (pi * R_a * R_s)

        Returns
        -------
        jax.numpy.array
            Returns the lensing potential for a mass with an elliptical dPIE
            convergence
        '''
        return super().function(x, y, **kwargs)

    def derivatives(self, x, y, e1, e2, center_x=0, center_y=0, **kwargs):
        '''Returns the deflection angles for a mass with an elliptical  dPIE
        convergence

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
        R_a : float
            core radius
        kappa_0 : float, keyword
            amplitude of the convergence, must be given as a keyword.
            This is related to the amplitude of the mass profile by
            rho_0 = kappa_0 * Critical_Surface_Density * (R_a + R_s) / (pi * R_a * R_s)

        Returns
        -------
        jax.numpy.array
            Returns the deflection angles for a mass with an elliptical dPIE
            convergence
        '''
        return super().derivatives(x, y, e1, e2, center_x, center_y, **kwargs)

    def hessian(self, x, y, e1, e2, center_x=0, center_y=0, **kwargs):
        '''Returns the hessian with respect to position for a mass with an
        elliptical dPIE convergence

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
        R_a : float
            core radius
        kappa_0 : float, keyword
            amplitude of the convergence, must be given as a keyword.
            This is related to the amplitude of the mass profile by
            rho_0 = kappa_0 * Critical_Surface_Density * (R_a + R_s) / (pi * R_a * R_s)

        Returns
        -------
        jax.numpy.array
            Returns the hessian with respect to position for a mass with an
            elliptical dPIE convergence
        '''
        return super().hessian(x, y, e1, e2, center_x, center_y, **kwargs)
