# Jax-Lensing-Profiles

Additional Jax friendly mass and light profiles for gravitational lensing.

While these profiles were originally designed for use with [Herculens](https://github.com/Herculens/herculens), they should be general enough to be used in other Jax based modeling software.

## Light Profiles
- `multi_gaussian_light.MultiGaussian`: Sum of multiple (vectorized) circular Gaussian profiles
- `multi_gaussian_light.MultiGaussianEllipse`: Sum of multiple (vectorized) elliptical Gaussian profiles

## Mass Profiles
- `gaussian_kappa_jax.GaussianKappa`: Single circular Gaussian convergence profile
- `gaussian_ellipse_kappa_jax.GaussianEllipseKappa`: Single elliptical Gaussian convergence profile
- `multi_gaussian_ellipse_kappa_jax.MultiGaussianEllipseKappa`: Sum of multiple (vectorized) elliptical Gaussian convergence profiles
- `MGE_jax.MGE`: A multi-gaussian-expansion of a given radial convergence profile
- `NFW.NFW`: circular NFW profile
- `NFW.TNFW`: circular Truncated NFW profile

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
