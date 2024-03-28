"""ritas."""

import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


@dataclass(frozen=True)
class DefaultColNames:
    """Mapping of where to find data within dataframes."""

    DISTANCE: str = "d"
    MASS: str = "mass"
    SWATH: str = "swath"


ColNames = DefaultColNames()

try:
    __version__ = version("ritas")
    pkgdir = Path(os.path.realpath(__file__)).parent.parent
    if not str(pkgdir).endswith("site-packages"):
        __version__ += "-dev"
except PackageNotFoundError:
    # package is not installed
    __version__ = "dev"
