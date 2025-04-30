import jax.numpy as jnp
import jax

from jax_lensing_profiles.Utility.f_function_jax import F, H
from jax.tree_util import Partial as partial

# References:
# https://arxiv.org/pdf/0705.0682.pdf
# https://arxiv.org/pdf/1101.0650.pdf
# https://arxiv.org/pdf/astro-ph/0102341

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
        returns TNFW lens potential
        """
        return self.__class__.psi(x, y, *args, **kwargs)

    def derivatives(self, x, y, *args, **kwargs):
        '''
        Use vectorized auto differentiation for the grad
        '''
        # use jnp.vectorized rather than jax.vmap so that all
        # dimensions of `x` and `y` are treated as batch dimensions
        # exclude up to `7` to account for the max number of params
        # between NFW and tNFW
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
        # exclude up to `7` to account for the max number of params
        # between NFW and tNFW
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
    this class contains function to evaluate the NFW derivatives
    '''
    param_names = ['ks', 'rs', 'center_x', 'center_y']
    lower_limit_default = {'ks': 1e-5, 'rs': 1e-5, 'center_x': -5, 'center_y': -5}
    upper_limit_default = {'ks': 1e+5, 'rs': 1e+5, 'center_x': 5, 'center_y': 5}
    fixed_default = {key: False for key in param_names}

    @staticmethod
    def center_and_scale(x, y, rs, center_x, center_y):
        xc = x - center_x
        yc = y - center_y
        norm = xc**2 + yc**2
        # add epsilon to the norm to prevent `nan` derivatives
        # jax requires this to be *inside* the sqrt to work with jax.grad
        r = jnp.sqrt(norm + 1e-10)
        xr = r / rs
        return xr

    @staticmethod
    def psi(x, y, ks, rs, center_x=0, center_y=0):
        """
        returns TNFW lens potential
        """
        xr = NFW.center_and_scale(x, y, rs, center_x, center_y)
        # Fx = NFW.F(xr)
        # Fx = F(xr)
        # psi = jnp.log(0.5 * xr)**2 - Fx * Fx * (1 - xr**2)
        psi = H(xr) + jnp.log(0.5 * xr)**2
        return 2.0 * ks * psi * rs**2

    @staticmethod
    def grad_stack(x, y, *args, **kwargs):
        return jnp.array(
            jax.grad(NFW.psi, argnums=(0, 1))(x, y, *args, **kwargs)
        )

    @staticmethod
    def part_grad_stack(ks, rs, center_x=0, center_y=0):
        return partial(
            NFW.grad_stack,
            ks=ks,
            rs=rs,
            center_x=center_x,
            center_y=center_y
        )

    @staticmethod
    def hessian_stack(x, y, *args, **kwargs):
        return jnp.array(
            jax.hessian(NFW.psi, argnums=(0, 1))(x, y, *args, **kwargs)
        )

    @staticmethod
    def part_hessian_stack(ks, rs, center_x=0, center_y=0):
        return partial(
            NFW.hessian_stack,
            ks=ks,
            rs=rs,
            center_x=center_x,
            center_y=center_y
        )


# ============================================================================================
# Class for tNFW
# ============================================================================================


class TNFW(NFWBase):
    """
    this class contains functions to evaluate TNFW derivatives
    """
    param_names = ['ks', 'rs', 'rt', 'center_x', 'center_y']
    lower_limit_default = {'ks': 1e-5, 'rs': 1e-5, 'rt': 1e-5, 'center_x': -5, 'center_y': -5}
    upper_limit_default = {'ks': 1e+5, 'rs': 1e+5, 'rt': 1e+5, 'center_x': 5, 'center_y': 5}
    fixed_default = {key: False for key in param_names}

    @staticmethod
    def center_and_scale(x, y, rs, rt, center_x, center_y):
        xc = x - center_x
        yc = y - center_y
        norm = xc**2 + yc**2
        # add epsilon to the norm to prevent `nan` derivatives
        # jax requires this to be *inside* the sqrt to work with jax.grad
        r = jnp.sqrt(norm + 1e-12)
        xr = r / rs
        tau = rt / rs
        return xr, tau

    @staticmethod
    def _psi_truncated_nfw(x, tau):
        # Fx = TNFW.F(x)
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
    def psi(x, y, ks, rs, rt, center_x=0, center_y=0):
        xr, tau = TNFW.center_and_scale(x, y, rs, rt, center_x, center_y)
        return 2.0 * ks * TNFW._psi_truncated_nfw(xr, tau) * rs**2

    @staticmethod
    def grad_stack(x, y, *args, **kwargs):
        return jnp.array(
            jax.grad(TNFW.psi, argnums=(0, 1))(x, y, *args, **kwargs)
        )

    @staticmethod
    def part_grad_stack(ks, rs, rt, center_x=0, center_y=0):
        return partial(
            TNFW.grad_stack,
            ks=ks,
            rs=rs,
            rt=rt,
            center_x=center_x,
            center_y=center_y
        )

    @staticmethod
    def hessian_stack(x, y, *args, **kwargs):
        return jnp.array(
            jax.hessian(TNFW.psi, argnums=(0, 1))(x, y, *args, **kwargs)
        )

    @staticmethod
    def part_hessian_stack(ks, rs, rt, center_x=0, center_y=0):
        return partial(
            TNFW.hessian_stack,
            ks=ks,
            rs=rs,
            rt=rt,
            center_x=center_x,
            center_y=center_y
        )
