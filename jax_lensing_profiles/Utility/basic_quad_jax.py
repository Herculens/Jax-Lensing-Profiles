import jax
import jax.numpy as jnp

from scipy.special import roots_legendre
from functools import partial


def nth_order_quad_base(func, a, b, args=(), kwargs={}, n=21):
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


@partial(jax.jit, static_argnums=(0, 5))
def vec_nth_order_quad(func, a, b, args=(), kwargs={}, n=21):
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
