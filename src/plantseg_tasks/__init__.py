"""Package description."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fractal-plantseg-tasks")
except PackageNotFoundError:
    __version__ = "uninstalled"


if version("plantseg") != "1.8.1":
    raise ImportError(
        "The version of plantseg_tasks is not compatible with the installed version of \
            plantseg. Please install the correct version of plantseg 1.8.3."
    )
