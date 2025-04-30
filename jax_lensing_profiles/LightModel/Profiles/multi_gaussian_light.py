"""This module defines Multi-Gaussian light profiles to compute the light
coming from a set of either circular or elliptical Gaussian profiles.  This
is a vectorized version of the single Gaussian light profile as defined in
Herculens.  That profile is based on the Lenstronomy implementation.

Copyright (c) 2025, herculens developers and contributors
Copyright (c) 2018, Simon Birrer & lenstronomy contributors
"""

__author__ = 'sibirrer', 'austinpeel', 'aymgal', 'ckrawczyk'


import numpy as np
import jax.numpy as jnp

from herculens.Util import param_util


__all__ = ['MultiGaussian', 'MultiGaussianEllipse']


class MultiGaussian(object):
    """
    class for a Multiple Gaussian light profile
    The two-dimensional Multiple Gaussian profile amplitude is defined such that the 2D integral leads to the 'sum(amp)' value.

    profile name in LightModel module: 'MULTI_GAUSSIAN'
    """
    param_names = ['amp', 'sigma', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'sigma': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 1000, 'sigma': 100, 'center_x': 100, 'center_y': 100}
    fixed_default = {key: False for key in param_names}

    def function(self, x, y, amp, sigma, center_x, center_y, reshape=None):
        """
        surface brightness per angular unit

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :param center_x: center of profile
        :param center_y: center of profile
        :return: surface brightness at (x, y)
        """
        reshape = (-1,) + (1,) * x.ndim
        amp_ = amp.reshape(reshape)
        sigma_ = sigma.reshape(reshape)
        center_x_ = center_x.reshape(reshape)
        center_y_ = center_y.reshape(reshape)
        c = (amp_ / (2 * np.pi * sigma_**2))
        R2 = (x - center_x_) ** 2 / sigma_**2 + (y - center_y_) ** 2 / sigma_**2
        return jnp.sum(c * jnp.exp(-R2 / 2.), axis=0)

    def total_flux(self, amp, sigma, center_x, center_y):
        """
        integrated flux of the profile

        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :param center_x: center of profile
        :param center_y: center of profile
        :return: total flux
        """
        return jnp.sum(amp)

    def light_3d(self, r, amp, sigma):
        """
        3D brightness per angular volume element

        :param r: 3d distance from center of profile
        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :return: 3D brightness per angular volume element
        """
        amp3d = amp / np.sqrt(2 * sigma**2) / np.sqrt(np.pi)
        sigma3d = sigma
        return self.function(r, 0, amp3d, sigma3d)


class MultiGaussianEllipse(object):
    """
    class for Multiple Gaussian light profile with ellipticity

    profile name in LightModel module: 'MULTI_GAUSSIAN_ELLIPSE'
    """
    param_names = ['amp', 'sigma', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'sigma': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 1000, 'sigma': 100, 'e1': -0.5, 'e2': -0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.multi_gaussian = MultiGaussian()

    def function(self, x, y, amp, sigma, e1, e2, center_x, center_y):
        """

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :param center_x: center of profile
        :param center_y: center of profile
        :return: surface brightness at (x, y)
        """
        reshape = (-1,) + (1,) * x.ndim
        amp_ = amp.reshape(reshape)
        sigma_ = sigma.reshape(reshape)
        e1_ = e1.reshape(reshape)
        e2_ = e2.reshape(reshape)
        center_x_ = center_x.reshape(reshape)
        center_y_ = center_y.reshape(reshape)
        x_, y_ = param_util.transform_e1e2_product_average(x, y, e1_, e2_, center_x_, center_y_)
        c = (amp_ / (2 * np.pi * sigma_**2))
        R2 = x_** 2 / sigma_**2 + y_** 2 / sigma_**2
        return jnp.sum(c * jnp.exp(-R2 / 2.), axis=0)

    def total_flux(self, amp, sigma=None, e1=None, e2=None, center_x=None, center_y=None):
        """
        total integrated flux of profile

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :param center_x: center of profile
        :param center_y: center of profile
        :return: total flux
        """
        return self.multi_gaussian.total_flux(amp, sigma, center_x, center_y)

    def light_3d(self, r, amp, sigma, e1=0, e2=0):
        """
        3D brightness per angular volume element

        :param r: 3d distance from center of profile
        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :return: 3D brightness per angular volume element
        """
        return self.multi_gaussian.light_3d(r, amp, sigma=sigma)
    