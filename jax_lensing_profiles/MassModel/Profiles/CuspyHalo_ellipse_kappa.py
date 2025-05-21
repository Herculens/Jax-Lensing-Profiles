"""This module defines a mass profile where the convergence follows 
and elliptical cuspy halo profile.  This makes use of the MGE profile.
The conventions of Keeton (2002, https://arxiv.org/pdf/astro-ph/0102341)
are used.
"""

__author__ = "WolfgangEnzi", "CKrawczyk", "astroskylee"

import jax.numpy as jnp

from .MGE import MGE


def CuspyHalo_3D_fn(r, R_s, kappa_s, gamma, n, **_):
    '''Radial scaled mass function for a cuspy halo profile

    Parameters
    ----------
    r : float
        radius
    R_s : float
        scale radius
    kappa_s : float
        amplitude of the convergence.  This is related to the amplitude
        of the mass profile by rho_s = kappa_s * Critical_Surface_Density / R_s
    gamma : float
        logarithmic slope at small radii
    n : float
        logarithmic slope at large radii

    Returns
    -------
    float
        scaled mass value for a 3D cuspy halo profile
    '''
    x = r / R_s
    t1 = x**gamma
    power = (n - gamma) / 2
    t2 = (1 + x**2)**power
    return kappa_s / (R_s * t1 * t2)


# Subclass MGE to initialize with the radial function
class CuspyHaloEllipseKappa(MGE):
    def __init__(self):
        super().__init__(
            CuspyHalo_3D_fn,
            'R_s',
            n_gauss=20,
            n_terms=28,
            sigma_start_mult=1/500,
            sigma_end_mult=20,
            three_d=True
        )

    # "override" these methods just to change the docstring
    def function(self, x, y, **kwargs):
        '''Returns the lensing potential for a mass with an elliptical cuspy
        halo convergence

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
        gamma : float
            logarithmic slope at small radii, must be given as a keyword
        n : float
            logarithmic slope at large radii, must be given as a keyword

        Returns
        -------
        jax.numpy.array
            Returns the lensing potential for a mass with an elliptical cuspy
            halo convergence
        '''
        return super().function(x, y, **kwargs)

    def derivatives(self, x, y, e1, e2, center_x=0, center_y=0, **kwargs):
        '''Returns the deflection angles for a mass with an elliptical cuspy
        halo convergence

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
        gamma : float
            logarithmic slope at small radii, must be given as a keyword
        n : float
            logarithmic slope at large radii, must be given as a keyword

        Returns
        -------
        jax.numpy.array
            Returns the deflection angles for a mass with an elliptical cuspy
            halo convergence
        '''
        return super().derivatives(x, y, e1, e2, center_x, center_y, **kwargs)

    def hessian(self, x, y, e1, e2, center_x=0, center_y=0, **kwargs):
        '''Returns the hessian with respect to position for a mass with an elliptical cuspy
        halo convergence

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
        gamma : float
            logarithmic slope at small radii, must be given as a keyword
        n : float
            logarithmic slope at large radii, must be given as a keyword

        Returns
        -------
        jax.numpy.array
            Returns the hessian with respect to position for a mass with an elliptical cuspy
            halo convergence
        '''
        return super().hessian(x, y, e1, e2, center_x, center_y, **kwargs)
