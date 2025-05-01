'''Helper functions for defining NFW mass profile'''

import jax.numpy as jnp

from jax import grad, custom_jvp


def H(r):
    # the typical piecewise way of writing this function can be combined
    # into a single `ln` that allows `sqrt(1-x**2)` to go complex for 
    # x > 1.  To avoid `nan` in the derivative when `r==1` a `where` is
    # used **inside** the `sqrt` function see:
    # https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
    # this can be `jax.grad` as many times as you want and still correctly
    # evaluate at `r==1`.
    #
    # note `r==0` is being avoided before this function is called 
    somr2 = jnp.sqrt((1 - jnp.where(r==1, r+1e-8, r)**2).astype(complex))
    t1 = jnp.log(r)
    t2 = jnp.log(1 + somr2)
    return -((t2 - t1)**2).real


# while F can be defined in terms of `grad(H)`, once complex inputs are used
# for MGEs it becomes harder to define `J` in a useful way
def F(r):
    # the typical piecewise way of writing this function can be combined
    # into a single `ln` that allows `sqrt(1-x**2)` to go complex for 
    # x > 1.  To avoid `nan` in the derivative when `r==1` a `where` is
    # used **inside** the `sqrt` function see:
    # https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
    # this can be `jax.grad` as many times as you want and still correctly
    # evaluate at `r==1`.
    #
    # note `r==0` is being avoided before this function is called 
    somr2 = jnp.sqrt((1 - jnp.where(r==1, r+1e-8, r)**2).astype(complex))
    t1 = jnp.log(r)
    t2 = 0.5 * jnp.log(2 * (1 + somr2) - r**2)
    return ((t2 - t1) / somr2).real



grad_F = jnp.vectorize(
    grad(F),
    signature='()->()'
)


def J(r):
    inv_r = 1 / r
    return inv_r * (grad_F(r) + inv_r)
