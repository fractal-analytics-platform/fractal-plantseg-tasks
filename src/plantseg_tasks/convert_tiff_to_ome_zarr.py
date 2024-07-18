"""This task converts simple H5 files to OME-Zarr."""

from typing import Optional

from fractal_tasks_core.utils import logger
from pydantic.v1 import Field
from pydantic.v1.decorator import validate_arguments

from plantseg_tasks.task_utils.converter_input_models import (
    CustomAxisInputModel,
    OMEZarrBuilderParams,
)


@validate_arguments
def convert_tiff_to_ome_zarr(
    input_path: str,
    output_dir: str,
    label_path: Optional[str] = None,
    custom_axis: CustomAxisInputModel = Field(
        title="Custom Axis", default_factory=CustomAxisInputModel
    ),
    ome_zarr_parameters: OMEZarrBuilderParams = Field(
        title="OME-Zarr Parameters", default_factory=OMEZarrBuilderParams
    ),
):
    """TIFF to OME-Zarr converter task.

    Args:
        input_path (str): Input path to the TIFF file,
            or a folder containing TIFF files.
        output_dir (str): Output path to save the OME-Zarr file.
        label_path (Optional[str]): Input path to the label TIFF file, or a folder
            containing TIFF files.
        custom_axis (list[AxisInputModel]): Custom axes to add to the OME-Zarr file.
            This field will override the default axes resolution and units found in the
            TIFF file.
        ome_zarr_parameters (OMEZarrBuilderParams): Parameters for the OME-Zarr builder.
    """
    pass


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=convert_tiff_to_ome_zarr,
        logger_name=logger.name,
    )
