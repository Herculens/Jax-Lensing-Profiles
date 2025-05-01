"""This module defines ``class GaussianEllipseKappa`` to compute the lensing properties
of an elliptical Gaussian profile with ellipticity in the convergence using the formulae
from Shajib (2019, https://academic.oup.com/mnras/article/488/1/1387/5526256).  This is
a JAX conversion of the Lenstronomy module.

Original author ajshajib.

Jax conversion author CKrawczyk.

Copyright (c) 2025, herculens developers and contributors
Copyright (c) 2018, Simon Birrer & lenstronomy contributors
"""

__author__ = "ajshajib", "CKrawczyk"

import jax.numpy as jnp
import herculens.Util.param_util as param_util

from jax_lensing_profiles.Utility.basic_quad_jax import nth_order_quad_base as quad
from jax_lensing_profiles.Utility.faddeeva_function import w_f
from jax.tree_util import Partial as partial


class GaussianEllipseKappa(object):
    """This class contains functions to evaluate the derivative and hessian matrix of
    the deflection potential for an elliptical Gaussian convergence.

    The formulae are from Shajib (2019, https://academic.oup.com/mnras/article/488/1/1387/5526256).
    """

    param_names = ["amp", "sigma", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "amp": 0,
        "sigma": 0,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 100,
        "sigma": 100,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
    }

    @staticmethod
    def center_and_scale(x, y, center_x, center_y, phi_g):
        x_shift = x - center_x
        y_shift = y - center_y
        y_shift = jnp.where(y_shift==0, y_shift+1e-20, y_shift)
        cos_phi = jnp.cos(phi_g)
        sin_phi = jnp.sin(phi_g)

        x_ = cos_phi * x_shift + sin_phi * y_shift
        y_ = -sin_phi * x_shift + cos_phi * y_shift
        return x_, y_, cos_phi, sin_phi

    def function(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        '''Potential values for a mass with a elliptical Gaussian convergence.

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
        e1 : float
            eccentricity modulus
        e2 : float
            eccentricity modulus
        center_x : float, optional
            center of profile, defaults to 0
        center_y : float, optional
            center of profile, defaults to 0

        Returns
        -------
        jax.numpy.array
            Potential values for a mass with a elliptical Gaussian convergence
        '''
        phi_g, q = param_util.ellipticity2phi_q(e1, e2)
        # adjusting amplitude to make the notation compatible with the
        # formulae given in Shajib (2019).
        amp_ = amp / (2 * jnp.pi * sigma**2)

        # converting ellipticity definition from q*x^2 + y^2/q to q^2*x^2 + y^2
        sigma_ = sigma * jnp.sqrt(q)  # * q

        x_, y_, _, _ = GaussianEllipseKappa.center_and_scale(
            x,
            y,
            center_x,
            center_y,
            phi_g
        )

        _b = 1.0 / (2.0 * sigma_**2)
        _p = jnp.sqrt(_b * q**2 / (1.0 - q**2))
        # _num_integral is vectorized over x_ and y_
        return jnp.vectorize(
            GaussianEllipseKappa._part_num_integral(amp_, sigma_, _p, q),
            signature='(),()->()'
        )(x_, y_)

    @staticmethod
    def pot_real_line_integrand(_x, _p, q):
        sig_func_re, _ = GaussianEllipseKappa.sigma_function(_p * _x, 0, q)
        alpha_x_ = sig_func_re
        return alpha_x_

    @staticmethod
    def pot_imag_line_integrand(_y, x_, _p, q):
        _, sig_func_im = GaussianEllipseKappa.sigma_function(_p * x_, _p * _y, q)
        alpha_y_ = sig_func_im
        return alpha_y_

    @staticmethod
    def _num_integral(x_, y_, amp_, sigma_, _p, q):
        factor = amp_ * sigma_ * jnp.sqrt(2 * jnp.pi / (1.0 - q**2))
        pot_on_real_line = quad(
            GaussianEllipseKappa.pot_real_line_integrand,
            0, x_,
            args=(_p, q),
            n=7
        )
        pot_on_imag_parallel = quad(
            GaussianEllipseKappa.pot_imag_line_integrand,
            0, y_,
            args=(x_, _p, q),
            n=7
        )
        return factor * (pot_on_real_line - pot_on_imag_parallel)

    @staticmethod
    def _part_num_integral(amp_, sigma_, _p, q):
        return partial(
            GaussianEllipseKappa._num_integral,
            amp_=amp_,
            sigma_=sigma_,
            _p=_p,
            q=q
        )

    def derivatives(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        '''Deflection angles for a mass with a elliptical Gaussian convergence.

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
        e1 : float
            eccentricity modulus
        e2 : float
            eccentricity modulus
        center_x : float, optional
            center of profile, defaults to 0
        center_y : float, optional
            center of profile, defaults to 0

        Returns
        -------
        jax.numpy.array
            Deflection angles for a mass with a elliptical Gaussian convergence
        '''
        phi_g, q = param_util.ellipticity2phi_q(e1, e2)
        # adjusting amplitude to make the notation compatible with the
        # formulae given in Shajib (2019).
        amp_ = amp / (2 * jnp.pi * sigma**2)

        # converting ellipticity definition from q*x^2 + y^2/q to q^2*x^2 + y^2
        sigma_ = sigma * jnp.sqrt(q)  # * q

        x_, y_, cos_phi, sin_phi = GaussianEllipseKappa.center_and_scale(
            x,
            y,
            center_x,
            center_y,
            phi_g
        )

        _p = q / sigma_ / jnp.sqrt(2 * (1.0 - q**2))

        sig_func_re, sig_func_im = GaussianEllipseKappa.sigma_function(_p * x_, _p * y_, q)

        factor = (
            amp_
            * sigma_
            * jnp.sqrt(2 * jnp.pi / (1.0 - q**2))
        )

        alpha_x_ = factor * sig_func_re
        alpha_y_ = -factor * sig_func_im

        # rotate back to the original frame
        f_x = alpha_x_ * cos_phi - alpha_y_ * sin_phi
        f_y = alpha_x_ * sin_phi + alpha_y_ * cos_phi

        return f_x, f_y

    def hessian(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        '''Hessian with respect to position for a mass with a elliptical Gaussian convergence.

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
        e1 : float
            eccentricity modulus
        e2 : float
            eccentricity modulus
        center_x : float, optional
            center of profile, defaults to 0
        center_y : float, optional
            center of profile, defaults to 0

        Returns
        -------
        jax.numpy.array
            Hessian with respect to position for a mass with a elliptical Gaussian convergence
        '''
        phi_g, q = param_util.ellipticity2phi_q(e1, e2)
        # adjusting amplitude to make the notation compatible with the
        # formulae given in Shajib (2019).
        amp_ = amp / (2 * jnp.pi * sigma**2)

        # converting ellipticity definition from q*x^2 + y^2/q to q^2*x^2 + y^2
        sigma_ = sigma * jnp.sqrt(q)  # * q

        x_, y_, cos_phi, sin_phi = GaussianEllipseKappa.center_and_scale(
            x,
            y,
            center_x,
            center_y,
            phi_g
        )

        _p = q / sigma_ / jnp.sqrt(2 * (1.0 - q**2))
        sig_func_re, sig_func_im = GaussianEllipseKappa.sigma_function(_p * x_, _p * y_, q)

        kappa = amp_ * jnp.exp(-(q**2 * x_**2 + y_**2) / 2 / sigma_**2)

        shear = (
            -1
            / (1 - q * q)
            * (
                (1 + q**2) * kappa
                - 2 * q * amp_
                + jnp.sqrt(2 * jnp.pi)
                * q
                * q
                * amp_
                * (x_ - 1j * y_)
                / sigma_
                / jnp.sqrt(1 - q * q)
                * (sig_func_re - 1j * sig_func_im)
            )
        )

        # in rotated frame
        f_xx_ = kappa + shear.real
        f_yy_ = kappa - shear.real
        f_xy_ = shear.imag

        # rotate back to the original frame
        f_xx = f_xx_ * cos_phi**2 + f_yy_ * sin_phi**2 - 2 * sin_phi * cos_phi * f_xy_
        f_yy = f_xx_ * sin_phi**2 + f_yy_ * cos_phi**2 + 2 * sin_phi * cos_phi * f_xy_
        f_xy = sin_phi * cos_phi * (f_xx_ - f_yy_) + (cos_phi**2 - sin_phi**2) * f_xy_

        return f_xx, f_yy, f_xy

    @staticmethod
    def sigma_function(x, y, q):
        y_sign = jnp.sign(y)
        y_ = y * y_sign

        w = w_f(x + 1j * y_)
        wq = w_f(q * x + 1j * y_ / q)

        # exponential factor in the 2nd term of eqn. (4.15) of Shajib (2019)
        exp_factor = jnp.exp(-x * x * (1 - q * q) - y_ * y_ * (1 / q / q - 1))

        sigma_func_real = w.imag - exp_factor * wq.imag
        sigma_func_imag = (-w.real + exp_factor * wq.real) * y_sign

        return sigma_func_real, sigma_func_imag
