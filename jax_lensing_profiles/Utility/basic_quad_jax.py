'''Gauss-Legendre quadrature written in JAX'''

import jax
import jax.numpy as jnp

from scipy.special import roots_legendre
from functools import partial


def nth_order_quad_base(func, a, b, args=(), kwargs={}, n=21):
    '''Nth order Gauss-Legendre quadrature

    Parameters
    ----------
    func : function
        function to integrate
    a : float
        lower bound of integration
    b : float
        upper bound of integration
    args : tuple, optional
        positional arguments of input function, by default ()
    kwargs : dict, optional
        keywords of input function, by default {}
    n : int, optional
        order of legendre roots to use, by default 21

    Returns
    -------
    float
        the integration of the input function between `a` and `b`
    '''
    # scipy.quad written in jax
    roots = jnp.array(roots_legendre(n)).T
    x_val = roots[:, 0:1]
    weights = roots[:, 1:2]
    # Integrate function with args from a to b
    aux = jnp.apply_along_axis(
        func,
        1,
        0.5 * ((b - a) * x_val + (b + a)),
        *args,
        **kwargs
    )
    res = 0.5 * (b - a) * jnp.sum(weights * aux, axis=0)
    return res[0]


nth_order_quad = jax.jit(nth_order_quad_base, static_argnums=(0, 5))


def log_segmented_nth_order_quad_base(
    func,
    a,
    b,
    args=(),
    kwargs={},
    n=21,
    segments=8,
    log_L=3.0,
):
    '''Nth order Gauss-Legendre quadrature on log-spaced sub-intervals

    Parameters
    ----------
    func : function
        function to integrate
    a : float
        lower bound of integration
    b : float
        upper bound of integration
    args : tuple, optional
        positional arguments of input function, by default ()
    kwargs : dict, optional
        keywords of input function, by default {}
    n : int, optional
        order of legendre roots to use on each segment, by default 21
    segments : int, optional
        number of log-spaced segments, by default 8
    log_L : float, optional
        Starting point of the logarithmic segmentation, e.g. log_L = 3 means
        the segment edges are built from np.logspace(-3, 0) before rescaling
        to [a, b], by default 3.0

    Returns
    -------
    float
        the segmented integration of the input function between `a` and `b`
    '''
    roots = jnp.array(roots_legendre(n)).T
    x_val = roots[:, 0]
    weights = roots[:, 1]

    u = jnp.logspace(-log_L, 0.0, segments + 1)
    edges = a + (b - a) * (u - u[0]) / (u[-1] - u[0])
    left, right = edges[:-1], edges[1:]

    x = 0.5 * ((right - left)[:, None] * x_val[None, :] + (right + left)[:, None])
    x_flat = x.reshape(-1, 1)

    aux = jnp.apply_along_axis(
        func,
        1,
        x_flat,
        *args,
        **kwargs
    )
    scale = 0.5 * (right - left)
    aux = aux.reshape((segments, n) + aux.shape[1:])
    weighted = jnp.tensordot(aux, weights, axes=([1], [0]))
    seg_int = weighted * scale.reshape((segments,) + (1,) * (weighted.ndim - 1))
    return jnp.squeeze(jnp.sum(seg_int, axis=0))


log_segmented_nth_order_quad = jax.jit(
    log_segmented_nth_order_quad_base,
    static_argnums=(0, 5, 6, 7),
)


@partial(jax.jit, static_argnums=(0, 5))
def vec_nth_order_quad(func, a, b, args=(), kwargs={}, n=21):
    '''Nth order Gauss-Legendre quadrature vectorized over the bounds

    Parameters
    ----------
    func : function
        function to integrate
    a : jax.numpy.array
        array of lower bounds of integration
    b : jax.numpy.array
        array of upper bounds of integration
    args : tuple, optional
        positional arguments of input function, by default ()
    kwargs : dict, optional
        keywords of input function, by default {}
    n : int, optional
        order of legendre roots to use, by default 21

    Returns
    -------
    jax.numpy.array
        the integration of the input function between each value in `a` and `b`
    '''
    part_nth_order_quad_base = partial(
        nth_order_quad_base,
        func=func,
        args=args,
        kwargs=kwargs,
        n=n
    )
    return jnp.vectorize(
        part_nth_order_quad_base,
        signature='(),()->()'
    )(a, b)
