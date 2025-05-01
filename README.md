# Jax-Lensing-Profiles

Additional Jax friendly mass and light profiles for gravitational lensing.

While these profiles were originally designed for use with [Herculens](https://github.com/Herculens/herculens), they should be general enough to be used in other Jax based modeling software.

## Light Profiles
- `multi_gaussian_light.MultiGaussian`: Sum of multiple (vectorized) circular Gaussian profiles
- `multi_gaussian_light.MultiGaussianEllipse`: Sum of multiple (vectorized) elliptical Gaussian profiles

## Mass Profiles
- `gaussian_kappa.GaussianKappa`: Single circular Gaussian convergence profile
- `gaussian_ellipse_kappa.GaussianEllipseKappa`: Single elliptical Gaussian convergence profile
- `multi_gaussian_ellipse_kappa.MultiGaussianEllipseKappa`: Sum of multiple (vectorized) elliptical Gaussian convergence profiles
- `MGE.MGE`: A multi-gaussian-expansion of a given radial convergence profile
- `NFW.NFW`: circular NFW profile
- `NFW.TNFW`: circular Truncated NFW profile
- `NFW_ellipse_kappa.NFWEllipseKappa`: Elliptical NFW convergence profile
- `Sersic_ellipse_kappa.SersicEllipseKappa`: Elliptical Sersic convergence profile

## Installation

```bash
git clone https://github.com/Herculens/Jax-Lensing-Profiles.git
cd Jax-Lensing-Profiles
pip install .
```

## Usage

All the included light and mass profiles are registered with Herculens on import

```python
import herculens
import jax_lensing_profiles

from herculens.LightModel.light_model_base import SUPPORTED_MODELS as SUPPORTED_LIGHT_MODELS
from herculens.MassModel.mass_model_base import SUPPORTED_MODELS as SUPPORTED_MASS_MODELS

print(SUPPORTED_LIGHT_MODELS)
print(SUPPORTED_MASS_MODELS)
```

## Defining new MGE profiles

If you want to use the multi-Gaussian-expansion class to create new mass profiles you will need the following:
- The radial profile for the convergence
- A subclass of the `MassModel.Profiles.MGE` class that overrides the `__init__` method to call the `MGE.__init__` with your target function

As and example this is the NFWEllipseKappa profile written in this way:

```python
import jax.numpy as jnp

from .MGE_jax import MGE
from jax_lensing_profiles.Utility.f_function_jax import J


def NFW_fn(r, R_s, kappa_s, **_):
    x = r / R_s
    return 2 * kappa_s * J(x)


class NFWEllipseKappa(MGE):
    def __init__(self):
        super().__init__(
            NFW_fn,
            'Rs',
            n_gauss=20,
            n_terms=28,
            sigma_start_mult=1/500,
            sigma_end_mult=20
        )
```

See `MassModel.Profiles.MGE` documentation for more details on the input parameters.
