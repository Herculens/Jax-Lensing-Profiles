# Jax-Lensing-Profiles

Additional Jax friendly mass and light profiles for gravitational lensing.

While these profiles were originally designed for use with [Herculens](https://github.com/Herculens/herculens), they should be general enough to be used in other Jax based modeling software.

## Light Profiles
- `multi_gaussian_light.MultiGaussian`: Sum of multiple (vectorized) circular Gaussian profiles [`MULTI_GAUSSIAN`]
- `multi_gaussian_light.MultiGaussianEllipse`: Sum of multiple (vectorized) elliptical Gaussian profiles [`MULTI_GAUSSIAN_ELLIPSE`]

## Mass Profiles
- `gaussian_kappa.GaussianKappa`: Single circular Gaussian convergence profile [`GAUSSIAN_KAPPA`]
- `gaussian_ellipse_kappa.GaussianEllipseKappa`: Single elliptical Gaussian convergence profile [`GAUSSIAN_ELLIPSE_KAPPA`]
- `multi_gaussian_ellipse_kappa.MultiGaussianEllipseKappa`: Sum of multiple (vectorized) elliptical Gaussian convergence profiles [`MULTI_GAUSSIAN_ELLIPSE_KAPPA`]
- `MGE.MGE`: A multi-gaussian-expansion of a given radial scaled mass profile
- `NFW.NFW`: circular NFW profile [`NFW`]
- `NFW.TNFW`: circular Truncated NFW profile [`TNFW`]
- `NFW_ellipse_kappa.NFWEllipseKappa`: Elliptical NFW convergence profile [`NFW_ELLIPSE_KAPPA`]
- `TNFW_ellipse_kappa.TNFWEllipseKappa`: Elliptical Truncated NFW convergence profile [`TNFW_ELLIPSE_KAPPA`]
- `CuspyNFW_ellipse_kappa.CuspyNFWEllipseKappa`: Elliptical cuspy NFW convergence profile [`CUSPY_NFW_ELLIPSE_KAPPA`]
- `CuspyHalo_ellipse_kappa.CuspyNFWEllipseKappa`: Elliptical cuspy halo convergence profile [`CUSPY_HALO_ELLIPSE_KAPPA`]
- `Sersic_ellipse_kappa.SersicEllipseKappa`: Elliptical Sersic convergence profile [`SERSIC_ELLIPSE_KAPPA`]

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
- The radial profile for the mass profile divided by the critical surface density.  This can either be a 3D radial profile (e.g. useful for NFW) or the project 2D profile (e.g. useful for Sersic).
- A subclass of the `MassModel.Profiles.MGE` class that overrides the `__init__` method to call the `MGE.__init__` with your target function

As and example this is the NFWEllipseKappa profile written in this way:

```python
import jax.numpy as jnp

from .MGE_jax import MGE


def NFW_3D_fn(r, R_s, kappa_s, **_):
    x = r / R_s
    return kappa_s / (r * (1 + x)**2)


class NFWEllipseKappa(MGE):
    def __init__(self):
        super().__init__(
            NFW_3D_fn,
            'R_s',
            n_gauss=20,
            n_terms=28,
            sigma_start_mult=1/500,
            sigma_end_mult=20,
            three_d=True
        )
```

See `MassModel.Profiles.MGE` documentation for more details on the input parameters.
