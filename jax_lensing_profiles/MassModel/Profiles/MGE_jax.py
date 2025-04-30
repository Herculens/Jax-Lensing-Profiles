"""This module defines ``class MGE`` to compute the lensing properties any radial convergence
profile with ellipticity using Gaussian decomposition as defined in Shajib (2019).
This is a JAX conversion inspired by the Lenstronomy module.

Original author ajshajib.

Jax conversion author CKrawczyk.

Copyright (c) 2025, herculens developers and contributors
Copyright (c) 2018, Simon Birrer & lenstronomy contributors
"""

__author__ = "ajshajib", "CKrawczyk"

import jax
import jax.numpy as jnp
import herculens.Util.param_util as param_util

from jax_lensing_profiles.Utility.basic_quad_jax import nth_order_quad_base as quad
from jax_lensing_profiles.Utility.faddeeva_function import w_f
from functools import partial


class MGE(object):
    def __init__(
        self,
        radial_fn,
        scale_name,
        sigma_start_mult=1/100,
        sigma_end_mult=20,
        n_gauss=20,
        n_terms=28
    ):
        self.radial_fn = radial_fn
        self.n_gauss = n_gauss
        self.scale_name = scale_name
        self.sigma_start_mult = sigma_start_mult
        self.sigma_end_mult = sigma_end_mult
        
        # eq. 6 for fixed nodes and weights
        n = jnp.arange(0, 2 * n_terms + 1)
        self.chi = jnp.sqrt((2 * n_terms * jnp.log(10) / 3) + 2j * jnp.pi * n)
        i = jnp.arange(1, n_terms)
        xi_last = 0.5**n_terms
        # we need all the "n choose k" values as a list
        # calculate them directly rather than using the
        # typical factorial representation
        n_choose_k = jnp.cumprod(
            (n_terms + 1 - i) / i
        )
        # this builds the last half of the xi array in reverse
        xi_right = xi_last * (1 + jnp.cumsum(n_choose_k))
        xi = jnp.hstack([
            jnp.array([0.5]),
            jnp.ones(n_terms),
            xi_right[::-1],
            jnp.array([xi_last])
        ])

        self.root_two_pi = jnp.sqrt(2 * jnp.pi)
        self.eta_const = 2 * self.root_two_pi * 10**(n_terms / 3)
        self.eta_n = xi * (-1)**n

        self.w = jnp.hstack([
            jnp.array([0.5]),
            jnp.ones(self.n_gauss - 2),
            jnp.array([0.5])
        ])

    def decompose(self, **kwargs):
        # log spaced sigma values for the decomposition
        r_min = kwargs[self.scale_name] * self.sigma_start_mult
        r_max = kwargs[self.scale_name] * self.sigma_end_mult
        log_sigmas = jnp.linspace(
            jnp.log(r_min),
            jnp.log(r_max),
            self.n_gauss
        )
        d_log_sigma = log_sigmas[1] - log_sigmas[0]
        sigmas = jnp.exp(log_sigmas)
        r_eval = sigmas.reshape(-1, 1) * self.chi

        f_eval = self.eta_const * jnp.sum(
            self.eta_n * jnp.real(
                self.radial_fn(r_eval, **kwargs)
            ),
            axis=1
        )
        amps = self.w * f_eval * d_log_sigma / self.root_two_pi
        return amps, sigmas

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

    @staticmethod
    def _function(amp, sigma, x, y, e1, e2, center_x=0, center_y=0, **_):
        # take in a scalar amp, sigma, x, and y
        phi_g, q = param_util.ellipticity2phi_q(e1, e2)
        # converting ellipticity definition from q*x^2 + y^2/q to q^2*x^2 + y^2
        sigma_ = sigma * jnp.sqrt(q)

        x_, y_, _, _ = MGE.center_and_scale(
            x,
            y,
            center_x,
            center_y,
            phi_g
        )

        _b = 1.0 / (2.0 * sigma_**2)
        _p = jnp.sqrt(_b * q**2 / (1.0 - q**2))
        return MGE._num_integral(x_, y_, amp, sigma_, _p, q)
    
    @staticmethod
    def _part_function(x, y, **kwargs):
        return partial(
            MGE._function,
            x=x,
            y=y,
            **kwargs
        )
    
    @staticmethod
    def _v1_function(x, y, amps, sigmas, **kwargs):
        # vectorize over a list of amps and sigmas and sum the result
        return jnp.vectorize(
            MGE._part_function(x, y, **kwargs),
            signature='(),()->()'
        )(amps, sigmas).sum()
    
    @staticmethod
    def _part_v1_function(amps, sigmas, **kwargs):
        return partial(
            MGE._v1_function,
            amps=amps,
            sigmas=sigmas,
            **kwargs
        )

    def function(self, x, y, **kwargs):
        # vectorize over x and y
        amps, sigmas = self.decompose(**kwargs)
        return jnp.vectorize(
            MGE._part_v1_function(amps, sigmas, **kwargs),
            signature='(),()->()'
        )(x, y)

    @staticmethod
    def pot_real_line_integrand(_x, _p, q):
        sig_func_re, _ = MGE.sigma_function(_p * _x, 0, q)
        alpha_x_ = sig_func_re
        return alpha_x_

    @staticmethod
    def pot_imag_line_integrand(_y, x_, _p, q):
        _, sig_func_im = MGE.sigma_function(_p * x_, _p * _y, q)
        alpha_y_ = sig_func_im
        return alpha_y_

    @staticmethod
    def _num_integral(x_, y_, amp_, sigma_, _p, q):
        factor = amp_ * sigma_ * jnp.sqrt(2 * jnp.pi / (1.0 - q**2))
        pot_on_real_line = quad(
            MGE.pot_real_line_integrand,
            0, x_,
            args=(_p, q),
            n=7
        )
        pot_on_imag_parallel = quad(
            MGE.pot_imag_line_integrand,
            0, y_,
            args=(x_, _p, q),
            n=7
        )
        return factor * (pot_on_real_line - pot_on_imag_parallel)

    def _derivatives(self, x, y, e1, e2, center_x=0, center_y=0, **kwargs):
        # version of the function that works on scalar inputs of x and y
        # jnp.vectorize will be used to allow any shape array
        # jax.jacfwd will be used to calculate the hessian
        amps, sigmas = self.decompose(**kwargs)
        phi_g, q = param_util.ellipticity2phi_q(e1, e2)

        # unlike the single gaussian case don't scale the amps by 1/(2 pi sigma**2)

        # converting ellipticity definition from q*x^2 + y^2/q to q^2*x^2 + y^2
        sigmas_ = sigmas * jnp.sqrt(q)

        x_, y_, cos_phi, sin_phi = MGE.center_and_scale(
            x,
            y,
            center_x,
            center_y,
            phi_g
        )

        _p = q / sigmas_ / jnp.sqrt(2 * (1.0 - q**2))

        sig_func_re, sig_func_im = MGE.sigma_function(_p * x_, _p * y_, q)
        factor = amps * sigmas_ * jnp.sqrt(2 * jnp.pi / (1.0 - q**2))

        alpha_x_ = jnp.sum(factor * sig_func_re, axis=0)
        alpha_y_ = jnp.sum(-factor * sig_func_im, axis=0)

        # rotate back to the original frame
        f_x = alpha_x_ * cos_phi - alpha_y_ * sin_phi
        f_y = alpha_x_ * sin_phi + alpha_y_ * cos_phi
        return jnp.stack([f_x, f_y])
    
    def derivatives(self, x, y, e1, e2, center_x=0, center_y=0, **kwargs):
        part = partial(
            self._derivatives,
            e1=e1,
            e2=e2,
            center_x=center_x,
            center_y=center_y,
            **kwargs
        )
        f = jnp.vectorize(
            part,
            signature='(),()->(i)'
        )(x, y)
        return f[..., 0], f[..., 1]

    def _hessian(self, x, y, e1, e2, center_x=0, center_y=0, **kwargs):
        return jnp.stack(jax.jacfwd(
            self._derivatives,
            argnums=(0, 1)
        )(x, y, e1, e2, center_x=center_x, center_y=center_y, **kwargs))
    
    def hessian(self, x, y, e1, e2, center_x=0, center_y=0, **kwargs):
        part = partial(
            self._hessian,
            e1=e1,
            e2=e2,
            center_x=center_x,
            center_y=center_y,
            **kwargs
        )
        h = jnp.vectorize(
            part,
            signature='(),()->(i,i)'
        )(x, y)
        return h[..., 0, 0], h[..., 1, 1], h[..., 0, 1]

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
