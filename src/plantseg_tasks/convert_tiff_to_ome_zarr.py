"""This task converts simple H5 files to OME-Zarr."""

from typing import Optional

from fractal_tasks_core.utils import logger
from pydantic.v1 import Field
from pydantic.v1.decorator import validate_arguments

from plantseg_tasks.task_utils.converter_input_models import (
    VALID_IMAGE_LAYOUT,
    CustomAxisInputModel,
    OMEZarrBuilderParams,
)
from plantseg_tasks.task_utils.io import load_tiff_images


@validate_arguments
def convert_tiff_to_ome_zarr(
    zarr_urls: list[str],
    zarr_dir: str,
    image_path: str,
    image_layout: VALID_IMAGE_LAYOUT = "ZYX",
    label_path: Optional[str] = None,
    new_image_key: str = "raw",
    new_label_key: str = "label",
    custom_axis: CustomAxisInputModel = Field(
        title="Custom Axis", default_factory=CustomAxisInputModel
    ),
    ome_zarr_parameters: OMEZarrBuilderParams = Field(
        title="OME-Zarr Parameters", default_factory=OMEZarrBuilderParams
    ),
):
    """TIFF to OME-Zarr converter task.

    Args:
        zarr_urls (list[str]): List of URLs to the OME-Zarr files.
            Not used in this task.
        zarr_dir (str): Output path to save the OME-Zarr file.
        image_path (str): Input path to the TIFF file,
            or a folder containing TIFF files.
        image_layout (VALID_IMAGE_LAYOUT): The layout of the image data.
        label_path (Optional[str]): Input path to the label TIFF file. Folder containing
            TIFF files is not yet supported.
        new_image_key (str): New key for the image data to
            be stored in the OME-Zarr.
        new_label_key (str): New key for the label data to
            be stored in the OME-Zarr.
        custom_axis (list[AxisInputModel]): Custom axes to add to the OME-Zarr file.
            This field will override the default axes resolution and units found in the
            TIFF file.
        ome_zarr_parameters (OMEZarrBuilderParams): Parameters for the OME-Zarr builder.
    """
    image_dc = load_tiff_images(
        image_path=image_path,
        label_path=label_path,
        new_image_key=new_image_key,
        new_label_key=new_label_key,
        image_layout=image_layout,
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=convert_tiff_to_ome_zarr,
        logger_name=logger.name,
    )
