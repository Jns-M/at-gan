"""AT-GAN (Arbitrary Tabular Generative Adversarial Network.)"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("at-gan")
except PackageNotFoundError:
    __version__ = "0.0.0"