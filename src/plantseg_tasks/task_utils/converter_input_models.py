"""Pydantic models for the converter tasks."""

from pathlib import Path
from typing import Literal

from pydantic.v1 import BaseModel, Field  # , field_validator

ALLOWED_TIFF_EXTENSIONS = [".tiff", ".tif"]
ALLOWED_H5_EXTENSIONS = [".h5", ".hdf5"]

VALID_AXIS_NAMES = Literal["c", "z", "y", "x"]

VALID_AXIS_UNITS_TYPE = Literal[
    "angstrom",
    "attometer",
    "centimeter",
    "decimeter",
    "exameter",
    "femtometer",
    "foot",
    "gigameter",
    "hectometer",
    "inch",
    "kilometer",
    "megameter",
    "meter",
    "micrometer",
    "mile",
    "millimeter",
    "nanometer",
    "parsec",
    "petameter",
    "picometer",
    "terameter",
    "yard",
    "yoctometer",
    "yottameter",
    "zeptometer",
    "zettameter",
]


def _validate_input_path(file_path: str, mode: Literal["h5", "tiff"]) -> None:
    """Validate that input path.

    File is valid if:
        - exists
        - is a file with the correct extension
        - is a directory with at least one file with the correct extension
    """
    path = Path(file_path)

    # File must exist
    if not path.exists():
        raise FileNotFoundError(f"File {path.name} does not exist.")

    # If file is a file, it must be a H5 file or a TIFF file
    if path.is_file() and mode == "h5":
        if path.suffix not in ALLOWED_H5_EXTENSIONS:
            raise ValueError(
                f"File {path.name} is not a valid H5 file. \
                H5 file must have one of the following extensions:\
                {ALLOWED_H5_EXTENSIONS}"
            )
        return None

    if path.is_file() and mode == "tiff":
        if path.suffix not in ALLOWED_TIFF_EXTENSIONS:
            raise ValueError(
                f"File {path.name} is not a valid TIFF file. \
                TiffFile must have one of the following extensions:\
                {ALLOWED_TIFF_EXTENSIONS}"
            )
        return None

    # If file is a directory, it must contain at least one H5 file or TIFF file
    if path.is_dir() and mode == "h5":
        all_h5 = []
        for ext in ALLOWED_H5_EXTENSIONS:
            all_h5 += list(path.glob(f"*{ext}"))

        if len(all_h5) == 0:
            raise ValueError(f"Folder {path.name} does not contain any H5 files.")
        return None

    if path.is_dir() and mode == "tiff":
        all_tiff = []
        for ext in ALLOWED_TIFF_EXTENSIONS:
            all_tiff += list(path.glob(f"*{ext}"))

        if len(all_tiff) == 0:
            raise ValueError(f"Folder {path.name} does not contain any TIFF files.")
        return None


class AxisScaleModel(BaseModel):
    """Input model for the axis scale to be used in the conversion.

    Attributes:
        axis_name (str): The name of the axis, must be one of 'c', 'z', 'y', 'x'.
        scale (float): The scale is used to set the
            resolution of the axis. It must corresponds to the voxel size for that axis.
    """

    axis_name: VALID_AXIS_NAMES = "c"
    scale: float = Field(default=1.0, ge=0.0)


class CustomAxisInputModel(BaseModel):
    """Input model for the custom axis to be used in the conversion.

    Attributes:
        axis (list[AxisScaleModel]): The list of axis to be used in the conversion.
            The order of the axis in the list should be the same as the order of
            the axis in the image.
            Must be the same length as the number of axis in the image.
        spatial_units (VALID_AXIS_UNITS_TYPE): The spatial units of the axis.
        channel_names (list[str]): The list of channel names. Must be the same length
            as the number of channels in the image.

    """

    axis: list[AxisScaleModel] = Field(default_factory=list)
    spatial_units: VALID_AXIS_UNITS_TYPE = "micrometer"
    channel_names: list[str] = Field(default_factory=list)


class OMEZarrBuilderParams(BaseModel):
    """Parameters for the OME-Zarr builder.

    Attributes:
        number_multiscale: The number of multiscale
            levels to create. Default is 4.
        scaling_factor_XY: The factor to downsample the XY plane.
            Default is 2.0, meaning every layer is half the size over XY.
        scaling_factor_Z: The factor to downsample the Z plane.
            Default is 1.0, no scaling on Z.
        create_all_ome_axis: Whether to create all OME axis.
            Default is True, meaning that missing axis will be created
            with a sigleton dimension.
    """

    number_multiscale: int = Field(default=4, ge=0)
    scaling_factor_XY: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        title="Scaling Factor XY",
    )
    scaling_factor_Z: float = Field(default=1.0, ge=1.0, le=10.0)
    create_all_ome_axis: bool = True
