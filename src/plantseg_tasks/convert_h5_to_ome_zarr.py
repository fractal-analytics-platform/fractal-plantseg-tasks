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
from plantseg_tasks.task_utils.io import load_h5_images


@validate_arguments
def convert_h5_to_ome_zarr(
    zarr_urls: list[str],
    zarr_dir: str,
    input_path: str,
    image_key: str = "raw",
    image_layout: VALID_IMAGE_LAYOUT = "ZYX",
    label_key: Optional[str] = None,
    new_image_key: Optional[str] = None,
    new_label_key: Optional[str] = None,
    custom_axis: CustomAxisInputModel = Field(
        title="Custom Axis", default_factory=CustomAxisInputModel
    ),
    ome_zarr_parameters: OMEZarrBuilderParams = Field(
        title="OME-Zarr Parameters", default_factory=OMEZarrBuilderParams
    ),
):
    """H5 to OME-Zarr converter task.

    Args:
        zarr_urls (list[str]): List of URLs to the OME-Zarr files.
            Not used in this task.
        zarr_dir (str): Output path to save the OME-Zarr file.
        input_path (str): Input path to the H5 file, or a folder containing H5 files.
        image_key (str): The image key in the H5 file where the image is stored.
        image_layout (VALID_IMAGE_LAYOUT): The layout of the image data.
            Must be one of 'ZYX', 'YX', 'XY', 'CZYX', 'ZCYX'.
        label_key (Optional[str]): The label key in the H5 file
            where a label/segmentation is stored.
        new_image_key (Optional[str]): New key for the image data to
            be stored in the OME-Zarr. If not provided, the original key will be used.
        new_label_key (Optional[str]): New key for the label data to
            be stored in the OME-Zarr. If not provided, the original key will be used.
        custom_axis (list[AxisInputModel]): Custom axes to add to the OME-Zarr file.
            This field will override the default axes resolution and units found in the
            H5 file.
        ome_zarr_parameters (OMEZarrBuilderParams): Parameters for the OME-Zarr builder.

    """
    image_dc = load_h5_images(
        input_path=input_path,
        image_key=image_key,
        label_key=label_key,
        new_image_key=new_image_key,
        new_label_key=new_label_key,
        image_layout=image_layout,
    )

    # Setup OME-Zarr zarr_url
    # Create OME-Zarr


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=convert_h5_to_ome_zarr,
        logger_name=logger.name,
    )
