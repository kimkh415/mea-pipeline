"""MEA spike-train analysis pipeline."""
from importlib.metadata import PackageNotFoundError, version as _version

# must be a PEP 440 string: spikeinterface stores it in to_dict() and parses it on reload
try:
    __version__ = _version("mea-pipeline")
except PackageNotFoundError:            # running from a source tree, uninstalled
    __version__ = "0.0.0"

__all__ = ["__version__"]
