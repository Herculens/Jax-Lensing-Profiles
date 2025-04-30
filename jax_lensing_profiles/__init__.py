"""Jax friendly Mass and Light profiles for gravitational lensing"""

from . import LightModel
from . import MassModel
from . import Utility

__version__ = "1.0.0"


# Register the new mass and light profiles with herculens on import

from herculens.LightModel import light_model_base
from herculens.MassModel import mass_model_base

light_model_base.STRING_MAPPING['MULTI_GAUSSIAN'] = LightModel.Profiles.MultiGaussian
light_model_base.STRING_MAPPING['MULTI_GAUSSIAN_ELLIPSE'] = LightModel.Profiles.MultiGaussianEllipse
light_model_base.SUPPORTED_MODELS = list(light_model_base.STRING_MAPPING.keys())

mass_model_base.STRING_MAPPING['GAUSSIAN_ELLIPSE_KAPPA'] = MassModel.Profiles.GaussianEllipseKappa
mass_model_base.STRING_MAPPING['GAUSSIAN_KAPPA'] = MassModel.Profiles.GaussianKappa
mass_model_base.STRING_MAPPING['MULTI_GAUSSIAN_ELLIPSE_KAPPA'] = MassModel.Profiles.MultiGaussianEllipseKappa
mass_model_base.STRING_MAPPING['MGE'] = MassModel.Profiles.MGE
mass_model_base.STRING_MAPPING['NFW'] = MassModel.Profiles.NFW
mass_model_base.STRING_MAPPING['TNFW'] = MassModel.Profiles.TNFW
mass_model_base.STRING_MAPPING['SERSIC_ELLIPSE_KAPPA'] = MassModel.Profiles.SersicEllipseKappa
mass_model_base.SUPPORTED_MODELS = list(mass_model_base.STRING_MAPPING.keys())
