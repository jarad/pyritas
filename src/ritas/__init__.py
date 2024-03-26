"""ritas."""

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("ritas")
    pkgdir = Path(os.path.realpath(__file__)).parent.parent
    if not str(pkgdir).endswith("site-packages"):
        __version__ += "-dev"
except PackageNotFoundError:
    # package is not installed
    __version__ = "dev"
