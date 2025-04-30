# Jax-Lensing-Profiles

Additional Jax friendly mass and light profiles for gravitational lensing.

While these profiles were originally designed for use with [Herculens](https://github.com/Herculens/herculens), they should be general enough to be used in other Jax based modeling software.


## Light Profiles
- `multi_gaussian_light.MultiGaussian`: Sum of multiple (vectorized) elliptical Gaussian profiles

## Mass Profiles
- `gaussian_kappa_jax.GaussianKappa`: Single circular Gaussian convergence profile
- `gaussian_ellipse_kappa_jax.GaussianEllipseKappa`: Single elliptical Gaussian convergence profile
- `multi_gaussian_ellipse_kappa_jax.MultiGaussianEllipseKappa`: Sum of multiple (vectorized) elliptical Gaussian convergence profiles
- `MGE_jax.MGE`: A multi-gaussian-expansion of a given radial convergence profile
- `NFW.NFW`: circular NFW profile
- `NFW.TNFW`: circular Truncated NFW profile
