"""This module defines ``class GaussianKappa`` to compute the lensing properties
of a Gaussian profile in the convergence using the formulae from Shajib (2019).
This is a JAX conversion of the Lenstronomy module.

Original author ajshajib.

Jax conversion author CKrawczyk.

Copyright (c) 2025, herculens developers and contributors
Copyright (c) 2018, Simon Birrer & lenstronomy contributors
"""

__author__ = "ajshajib", "CKrawczyk"

import jax.numpy as jnp


def E1(x):
    """Abramowitz Stegun (1970) approximation of the exponential integral of
    the first order, E1.

    Args:
       x: input

    Returns:
       The exponential integral of the first order, E1(x)

    Reference https://github.com/HajimeKawahara/exojax/tree/master/src/exojax/special
    """
    A0 = -0.57721566
    A1 = 0.99999193
    A2 = -0.24991055
    A3 = 0.05519968
    A4 = -0.00976004
    A5 = 0.00107857
    B1 = 8.5733287401
    B2 = 18.059016973
    B3 = 8.6347608925
    B4 = 0.2677737343
    C1 = 9.5733223454
    C2 = 25.6329561486
    C3 = 21.0996530827
    C4 = 3.9584969228

    x2 = x**2
    x3 = x**3
    x4 = x**4
    x5 = x**5
    ep1A = -jnp.log(x) + A0 + A1 * x + A2 * x2 + A3 * x3 + A4 * x4 + A5 * x5
    num = jnp.exp(-x) * (x4 + B1 * x3 + B2 * x2 + B3 * x + B4)
    denom = x5 + C1 * x4 + C2 * x3 + C3 * x2 + C4 * x
    ep1B = num / denom
    ep = jnp.where(x <= 1.0, ep1A, ep1B)
    return ep


class GaussianKappa(object):
    param_names = ["amp", "sigma", "center_x", "center_y"]
    lower_limit_default = {"amp": 0, "sigma": 0, "center_x": -100, "center_y": -100}
    upper_limit_default = {"amp": 100, "sigma": 100, "center_x": 100, "center_y": 100}

    @staticmethod
    def center(x, y, center_x, center_y):
        xc = x - center_x
        yc = y - center_y
        norm = xc**2 + yc**2
        # add epsilon to the norm to prevent `nan` derivatives
        # jax requires this to be *inside* the sqrt to work with jax.grad
        R = jnp.sqrt(norm + 1e-10)
        return xc, yc, R

    @staticmethod
    def _amp2d_to_3d(amp, sigma_x, sigma_y):
        """Converts 3d density into 2d density parameter.

        :param amp:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        return amp / (jnp.sqrt(jnp.pi) * jnp.sqrt(sigma_x * sigma_y * 2))

    @staticmethod
    def mass_2d(R, amp, sigma):
        """
        :param R:
        :param amp:
        :param sigma:
        :return:
        """
        sigma_x, sigma_y = sigma, sigma
        amp2d = amp / (jnp.sqrt(jnp.pi) * jnp.sqrt(sigma_x * sigma_y * 2))
        c = 1.0 / (2 * sigma_x * sigma_y)
        return amp2d * 2 * jnp.pi * 1.0 / (2 * c) * (1.0 - jnp.exp(-c * R**2))

    @staticmethod
    def alpha_abs(R, amp, sigma):
        amp_density = GaussianKappa._amp2d_to_3d(amp, sigma, sigma)
        alpha = GaussianKappa.mass_2d(R, amp_density, sigma) / jnp.pi / R
        return alpha

    def function(self, x, y, amp, sigma, center_x=0, center_y=0):
        '''Potential values for a mass with a circular Gaussian convergence.

        Parameters
        ----------
        x : jax.numpy.array
            coordinate on the sky
        y : jax.numpy.array
            coordinate on the sky
        amp : float
            amplitudes, such that 2D integral leads to this value
        sigma : float
            sigma of Gaussian
        center_x : float, optional
            center of profile, defaults to 0
        center_y : float, optional
            center of profile, defaults to 0

        Returns
        -------
        jax.numpy.array
            Potential values for a mass with a circular Gaussian convergence
        '''
        _, _, R = GaussianKappa.center(x, y, center_x, center_y)
        c = 1.0 / (2 * sigma**2)
        value = c * R**2
        integral_term = 0.5 * (jnp.euler_gamma + jnp.log(value) + E1(value))
        amp_density = GaussianKappa._amp2d_to_3d(amp, sigma, sigma)
        amp2d = amp_density / (jnp.sqrt(jnp.pi) * jnp.sqrt(2) * sigma)
        amp2d *= 2 * 1.0 / (2 * c)
        return integral_term * amp2d

    def derivatives(self, x, y, amp, sigma, center_x=0, center_y=0):
        '''Deflection angles for a mass with a circular Gaussian convergence.

        Parameters
        ----------
        x : jax.numpy.array
            coordinate on the sky
        y : jax.numpy.array
            coordinate on the sky
        amp : float
            amplitudes, such that 2D integral leads to this value
        sigma : float
            sigma of Gaussian
        center_x : float, optional
            center of profile, defaults to 0
        center_y : float, optional
            center of profile, defaults to 0

        Returns
        -------
        jax.numpy.array
            Deflection angles for a mass with a circular Gaussian convergence
        '''
        xc, yc, R = GaussianKappa.center(x, y, center_x, center_y)
        alpha = GaussianKappa.alpha_abs(R, amp, sigma)
        return alpha / R * xc, alpha / R * yc

    @staticmethod
    def d_alpha_dr(R, amp, sigma_x, sigma_y):
        """

        :param R:
        :param amp:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        c = 1.0 / (2 * sigma_x * sigma_y)
        A = GaussianKappa._amp2d_to_3d(amp, sigma_x, sigma_y) * jnp.sqrt(
            2 / jnp.pi * sigma_x * sigma_y
        )
        return 1.0 / R**2 * (-1 + (1 + 2 * c * R**2) * jnp.exp(-c * R**2)) * A

    def hessian(self, x, y, amp, sigma, center_x=0, center_y=0):
        '''Hessian with respect to position for a mass with a circular Gaussian convergence.

        Parameters
        ----------
        x : jax.numpy.array
            coordinate on the sky
        y : jax.numpy.array
            coordinate on the sky
        amp : float
            amplitudes, such that 2D integral leads to this value
        sigma : float
            sigma of Gaussian
        center_x : float, optional
            center of profile, defaults to 0
        center_y : float, optional
            center of profile, defaults to 0

        Returns
        -------
        jax.numpy.array
            Hessian with respect to position for a mass with a circular Gaussian convergence
        '''
        xc, yc, R = GaussianKappa.center(x, y, center_x, center_y)
        alpha = GaussianKappa.alpha_abs(R, amp, sigma)
        d_alpha_dr = -GaussianKappa.d_alpha_dr(R, amp, sigma, sigma)
        f_xx = -(d_alpha_dr / R + alpha / R**2) * xc**2 / R + alpha / R
        f_yy = -(d_alpha_dr / R + alpha / R**2) * yc**2 / R + alpha / R
        f_xy = -(d_alpha_dr / R + alpha / R**2) * xc * yc / R
        return f_xx, f_yy, f_xy
