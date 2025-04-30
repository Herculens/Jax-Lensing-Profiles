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


# grad_H = jnp.vectorize(
#     grad(H),
#     signature='()->()'
# )


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


# def F(r):
#     return 0.5 * r * grad_H(r)


grad_F = jnp.vectorize(
    grad(F),
    signature='()->()'
)


def J(r):
    inv_r = 1 / r
    return inv_r * (grad_F(r) + inv_r)


# def F_greater_1(x):
#     term = jnp.sqrt(x**2 - 1)
#     return jnp.arctan(term) / term


# def F_less_1(x):
#     # equivalent to jnp.arctanh(jnp.sqrt(1 - x**2)) / jnp.sqrt(1 - x**2)
#     # but more stable for small values of x
#     term = jnp.sqrt(1 - x**2)
#     return (0.5 * jnp.log(2 + 2 * term - x**2) - jnp.log(x)) / term


# def F_equal_1(x):
#     # the expansion for either case is the same near 1
#     # up to the 2nd order term.
#     # replace it with smooth function of x so that the
#     # first and second derivatives of the piecewise is
#     # continuous at 1.
#     xm1 = x - 1
#     return 1 - (2/3) * xm1 + (7/15) * xm1**2


# @custom_jvp
# def F(x):
#     output = jnp.empty_like(x)
#     output = jnp.where(
#         (x > 0) & (x <= 0.99999),
#         F_less_1(x),
#         output
#     )
#     output = jnp.where(
#         x >= 1.00001,
#         F_greater_1(x),
#         output
#     )
#     output = jnp.where(
#         (x > 0.99999) & (x < 1.00001),
#         F_equal_1(x),
#         output
#     )
#     return output


# @F.defjvp
# def F_jvp(primals, tangents):
#     # define a custom jvp to avoid the issue using `jnp.where` with `jax.grad`
#     x, = primals
#     x_dot, = tangents

#     near_one = (x > 0.99999) & (x < 1.00001)
#     primal_out = F(x)
#     # masked_out = F(jnp.where(near_one, -x, x))
#     # tangent_out = (x * masked_out - 1/x) / (1 - x**2)

#     tangent_out = jnp.empty_like(x)
#     tangent_out = jnp.where(
#         near_one,
#         -2/3 + (14/15) * (x - 1),
#         (x * primal_out - 1/x) / (1 - x**2)
#     )
#     return primal_out, x_dot * tangent_out


# @custom_jvp
# def H(x):
#     xm1 = x - 1
#     output = jnp.empty_like(x)
#     output = jnp.where(
#         x <= 0.99999,
#         -jnp.arccosh(1 / x)**2,
#         output
#     )
#     output = jnp.where(
#         x >= 1.00001,
#         jnp.arccos(1 / x)**2,
#         output
#     )
#     output = jnp.where(
#         (x > 0.99999) & (x < 1.00001),
#         2 * xm1 - (5/3) * xm1**2 + (64/45) * xm1**3 - (26/21) * xm1**4,
#         output
#     )
#     return output


# @H.defjvp
# def H_jvp(primals, tangents):
#     x, = primals
#     x_dot, = tangents
#     primal_out = H(x)
#     tangent_out = 2 * F(x) / x
#     return primal_out, x_dot * tangent_out
