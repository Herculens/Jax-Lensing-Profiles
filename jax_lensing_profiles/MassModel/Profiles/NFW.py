"""This module defines a mass profile where the convergence follows 
and circular NFW profile or truncated-NFW function.  The conventions
of Keeton (2002, https://arxiv.org/pdf/astro-ph/0102341) are used.

Other references used:
Baltz et al. (2008, https://arxiv.org/pdf/0705.0682)
Oguri et al. (2011, https://arxiv.org/pdf/1101.0650)
"""

__author__ = "WolfgangEnzi", "CKrawczyk"

import jax.numpy as jnp
import jax

from jax_lensing_profiles.Utility.f_function_jax import F, H
from jax.tree_util import Partial as partial


# ============================================================================================
# Base class for NFW and tNFW
# ============================================================================================


class NFWBase(object):
    '''
    Bass class for the NFW and TNFW classes with all shared methods
    '''
    @staticmethod
    def psi(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def grad_stack(x, y, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def hessian_stack(x, y, *args, **kwargs):
        raise NotImplementedError

    def function(self, x, y, *args, **kwargs):
        """
        returns NFW lens potential
        """
        return self.__class__.psi(x, y, *args, **kwargs)

    def derivatives(self, x, y, *args, **kwargs):
        '''
        Use vectorized auto differentiation for the grad
        '''
        # use jnp.vectorized rather than jax.vmap so that all
        # dimensions of `x` and `y` are treated as batch dimensions
        grad_vec = jnp.vectorize(
            self.__class__.part_grad_stack(*args, **kwargs),
            signature='(),()->(i)'
        )(x, y)
        return grad_vec[..., 0], grad_vec[..., 1]

    def hessian(self, x, y, *args, **kwargs):
        '''
        Use vectorized auto hessian
        '''
        # use jnp.vectorized rather than jax.vmap so that all
        # dimensions of `x` and `y` are treated as batch dimensions
        h = jnp.vectorize(
            self.__class__.part_hessian_stack(*args, **kwargs),
            signature='(),()->(i,i)'
        )(x, y)
        return h[..., 0, 0], h[..., 1, 1], h[..., 0, 1]

# ============================================================================================
# Class for NFW
# ============================================================================================


class NFW(NFWBase):
    '''
    This class contains function to evaluate the NFW derivatives
    '''
    param_names = ['kappa_s', 'R_s', 'center_x', 'center_y']
    lower_limit_default = {'kappa_s': 1e-5, 'R_s': 1e-5, 'center_x': -5, 'center_y': -5}
    upper_limit_default = {'kappa_s': 1e+5, 'R_s': 1e+5, 'center_x': 5, 'center_y': 5}
    fixed_default = {key: False for key in param_names}

    @staticmethod
    def center_and_scale(x, y, R_s, center_x, center_y):
        xc = x - center_x
        yc = y - center_y
        norm = xc**2 + yc**2
        # add epsilon to the norm to prevent `nan` derivatives
        # jax requires this to be *inside* the sqrt to work with jax.grad
        r = jnp.sqrt(norm + 1e-10)
        xr = r / R_s
        return xr

    @staticmethod
    def psi(x, y, kappa_s, R_s, center_x=0, center_y=0):
        """
        returns tNFW lens potential
        """
        xr = NFW.center_and_scale(x, y, R_s, center_x, center_y)
        # Fx = NFW.F(xr)
        # Fx = F(xr)
        # psi = jnp.log(0.5 * xr)**2 - Fx * Fx * (1 - xr**2)
        psi = H(xr) + jnp.log(0.5 * xr)**2
        return 2.0 * kappa_s * psi * R_s**2

    @staticmethod
    def grad_stack(x, y, *args, **kwargs):
        return jnp.array(
            jax.grad(NFW.psi, argnums=(0, 1))(x, y, *args, **kwargs)
        )

    @staticmethod
    def part_grad_stack(kappa_s, R_s, center_x=0, center_y=0):
        return partial(
            NFW.grad_stack,
            kappa_s=kappa_s,
            R_s=R_s,
            center_x=center_x,
            center_y=center_y
        )

    @staticmethod
    def hessian_stack(x, y, *args, **kwargs):
        return jnp.array(
            jax.hessian(NFW.psi, argnums=(0, 1))(x, y, *args, **kwargs)
        )

    @staticmethod
    def part_hessian_stack(kappa_s, R_s, center_x=0, center_y=0):
        return partial(
            NFW.hessian_stack,
            kappa_s=kappa_s,
            R_s=R_s,
            center_x=center_x,
            center_y=center_y
        )

    # "override" these methods just to change the docstring
    def function(self, x, y, *args, **kwargs):
        '''Returns the lensing potential for a mass with an circular NFW convergence

        Parameters
        ----------
        x : jax.numpy.array
            coordinate on the sky
        y : jax.numpy.array
            coordinate on the sky
        center_x : float, keyword
            center of the profile, must be given as a keyword
        center_y : float, keyword
            center of the profile, must be given as a keyword
        kappa_s : float, keyword
            amplitude of the convergence, must be given as a keyword
        R_s : float, keyword
            scale radius, must be given as a keyword

        Returns
        -------
        jax.numpy.array
            Returns the lensing potential for a mass with an circular NFW convergence
        '''
        return super().function(x, y, *args, **kwargs)

    def derivatives(self, x, y, *args, **kwargs):
        '''Returns deflection angles for a mass with an circular NFW convergence

        Parameters
        ----------
        x : jax.numpy.array
            coordinate on the sky
        y : jax.numpy.array
            coordinate on the sky
        center_x : float, keyword
            center of the profile, must be given as a keyword
        center_y : float, keyword
            center of the profile, must be given as a keyword
        kappa_s : float, keyword
            amplitude of the convergence, must be given as a keyword
        R_s : float, keyword
            scale radius, must be given as a keyword

        Returns
        -------
        jax.numpy.array
            Returns deflection angles for a mass with an circular NFW convergence
        '''
        return super().derivatives(x, y, *args, **kwargs)
    
    def hessian(self, x, y, *args, **kwargs):
        '''Returns the hessian with respect to position for a mass with an circular NFW convergence

        Parameters
        ----------
        x : jax.numpy.array
            coordinate on the sky
        y : jax.numpy.array
            coordinate on the sky
        center_x : float, keyword
            center of the profile, must be given as a keyword
        center_y : float, keyword
            center of the profile, must be given as a keyword
        kappa_s : float, keyword
            amplitude of the convergence, must be given as a keyword
        R_s : float, keyword
            scale radius, must be given as a keyword

        Returns
        -------
        jax.numpy.array
            Returns the hessian with respect to position for a mass with an circular NFW convergence
        '''
        return super().hessian(x, y, *args, **kwargs)


# ============================================================================================
# Class for tNFW
# ============================================================================================


class TNFW(NFWBase):
    """
    This class contains functions to evaluate tNFW derivatives
    """
    param_names = ['kappa_s', 'R_s', 'R_t', 'center_x', 'center_y']
    lower_limit_default = {'kappa_s': 1e-5, 'R_s': 1e-5, 'R_t': 1e-5, 'center_x': -5, 'center_y': -5}
    upper_limit_default = {'kappa_s': 1e+5, 'R_s': 1e+5, 'R_t': 1e+5, 'center_x': 5, 'center_y': 5}
    fixed_default = {key: False for key in param_names}

    @staticmethod
    def center_and_scale(x, y, R_s, R_t, center_x, center_y):
        xc = x - center_x
        yc = y - center_y
        norm = xc**2 + yc**2
        # add epsilon to the norm to prevent `nan` derivatives
        # jax requires this to be *inside* the sqrt to work with jax.grad
        r = jnp.sqrt(norm + 1e-12)
        xr = r / R_s
        tau = R_t / R_s
        return xr, tau

    @staticmethod
    def _psi_truncated_nfw(x, tau):
        Fx = F(x)
        x2 = x**2
        tau2 = tau**2
        sx2tau2 = jnp.sqrt(tau2 + x2)
        logx = jnp.log(x)
        Lx = logx - jnp.log(sx2tau2 + tau)
        tau2m1 = tau2 - 1
        x2m1 = x2 - 1
        ltau = jnp.log(tau)
        log2 = jnp.log(2)
        A = (tau / (tau2 + 1))**2
        B = -2 * jnp.pi * sx2tau2 \
            + 4 * x2m1 * Fx \
            + tau2m1 * x2m1 * Fx**2 \
            + 2 * tau * (jnp.pi - jnp.pi * log2 + tau * log2) \
            + 2 * log2 - 2 * ltau \
            - 2 * ltau * (jnp.pi * tau + tau2 * (log2 - 1) - log2) \
            - tau2m1 * ltau**2 \
            + 2 * (tau2m1 * ltau - tau2 - 1) * logx \
            - 2 * sx2tau2 * (1.0 / tau - tau) * Lx \
            + 2 * jnp.pi * tau * (logx - Lx) \
            + tau2m1 * Lx * Lx
        return B * A

    @staticmethod
    def psi(x, y, kappa_s, R_s, R_t, center_x=0, center_y=0):
        xr, tau = TNFW.center_and_scale(x, y, R_s, R_t, center_x, center_y)
        return 2.0 * kappa_s * TNFW._psi_truncated_nfw(xr, tau) * R_s**2

    @staticmethod
    def grad_stack(x, y, *args, **kwargs):
        return jnp.array(
            jax.grad(TNFW.psi, argnums=(0, 1))(x, y, *args, **kwargs)
        )

    @staticmethod
    def part_grad_stack(kappa_s, R_s, R_t, center_x=0, center_y=0):
        return partial(
            TNFW.grad_stack,
            kappa_s=kappa_s,
            R_s=R_s,
            R_t=R_t,
            center_x=center_x,
            center_y=center_y
        )

    @staticmethod
    def hessian_stack(x, y, *args, **kwargs):
        return jnp.array(
            jax.hessian(TNFW.psi, argnums=(0, 1))(x, y, *args, **kwargs)
        )

    @staticmethod
    def part_hessian_stack(kappa_s, R_s, R_t, center_x=0, center_y=0):
        return partial(
            TNFW.hessian_stack,
            kappa_s=kappa_s,
            R_s=R_s,
            R_t=R_t,
            center_x=center_x,
            center_y=center_y
        )

    # "override" these methods just to change the docstring
    def function(self, x, y, *args, **kwargs):
        '''Returns the lensing potential for a mass with an circular tNFW convergence

        Parameters
        ----------
        x : jax.numpy.array
            coordinate on the sky
        y : jax.numpy.array
            coordinate on the sky
        center_x : float, keyword
            center of the profile, must be given as a keyword
        center_y : float, keyword
            center of the profile, must be given as a keyword
        kappa_s : float, keyword
            amplitude of the convergence, must be given as a keyword
        R_s : float, keyword
            scale radius, must be given as a keyword
        R_t : float, keyword
            truncation radius, must be given as a keyword

        Returns
        -------
        jax.numpy.array
            Returns the lensing potential for a mass with an circular tNFW convergence
        '''
        return super().function(x, y, *args, **kwargs)

    def derivatives(self, x, y, *args, **kwargs):
        '''Returns deflection angles for a mass with an circular tNFW convergence

        Parameters
        ----------
        x : jax.numpy.array
            coordinate on the sky
        y : jax.numpy.array
            coordinate on the sky
        center_x : float, keyword
            center of the profile, must be given as a keyword
        center_y : float, keyword
            center of the profile, must be given as a keyword
        kappa_s : float, keyword
            amplitude of the convergence, must be given as a keyword
        R_s : float, keyword
            scale radius, must be given as a keyword
        R_t : float, keyword
            truncation radius, must be given as a keyword

        Returns
        -------
        jax.numpy.array
            Returns deflection angles for a mass with an circular tNFW convergence
        '''
        return super().derivatives(x, y, *args, **kwargs)
    
    def hessian(self, x, y, *args, **kwargs):
        '''Returns the hessian with respect to position for a mass with an circular tNFW convergence

        Parameters
        ----------
        x : jax.numpy.array
            coordinate on the sky
        y : jax.numpy.array
            coordinate on the sky
        center_x : float, keyword
            center of the profile, must be given as a keyword
        center_y : float, keyword
            center of the profile, must be given as a keyword
        kappa_s : float, keyword
            amplitude of the convergence, must be given as a keyword
        R_s : float, keyword
            scale radius, must be given as a keyword
        R_t : float, keyword
            truncation radius, must be given as a keyword

        Returns
        -------
        jax.numpy.array
            Returns the hessian with respect to position for a mass with an circular tNFW convergence
        '''
        return super().hessian(x, y, *args, **kwargs)
