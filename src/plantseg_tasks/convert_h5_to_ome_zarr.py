"""This task converts simple H5 files to OME-Zarr."""

from pathlib import Path
from typing import Optional

from fractal_tasks_core.utils import logger
from pydantic import Field, validate_call

from plantseg_tasks.task_utils.converter_input_models import (
    ALLOWED_H5_EXTENSIONS,
    VALID_IMAGE_LAYOUT,
    CustomAxisInputModel,
    OMEZarrBuilderParams,
)
from plantseg_tasks.task_utils.io import (
    correct_image_metadata,
    create_ome_zarr,
    load_h5_images,
)


def convert_single_h5_to_ome(
    zarr_dir: str,
    input_path: str,
    image_key: str = "raw",
    image_layout: VALID_IMAGE_LAYOUT = "ZYX",
    label_key: Optional[str] = None,
    new_image_key: Optional[str] = None,
    new_label_key: Optional[str] = None,
    custom_axis: CustomAxisInputModel = Field(
        title="Custom Axis", default=CustomAxisInputModel()
    ),
    ome_zarr_parameters: OMEZarrBuilderParams = Field(
        title="OME-Zarr Parameters", default=OMEZarrBuilderParams()
    ),
):
    """H5 to OME-Zarr converter task.

    Args:
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
    image_ds = load_h5_images(
        input_path=input_path,
        image_key=image_key,
        label_key=label_key,
        image_layout=image_layout,
        new_image_key=new_image_key,
        new_label_key=new_label_key,
    )
    logger.info(f"Loaded image from {input_path}")

    zarr_url = Path(zarr_dir) / f"{Path(input_path).stem}.zarr"
    image_ds = correct_image_metadata(image_ds, custom_axis=custom_axis)
    logger.info(f"Corrected metadata for {input_path}")
    return create_ome_zarr(
        zarr_url=zarr_url,
        path=image_ds.image_key,
        name=image_ds.image_key,
        image=image_ds,
        omezarr_params=ome_zarr_parameters,
    )


@validate_call
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
    if not Path(input_path).exists():
        raise ValueError("Input path does not exist.")

    if Path(input_path).is_dir():
        files = []
        for ext in ALLOWED_H5_EXTENSIONS:
            files += list(Path(input_path).glob(f"*{ext}"))

        if len(files) == 0:
            raise ValueError(f"Folder {input_path} does not contain any H5 files.")

        logger.info(
            f"Converting from directory: {input_path}. Found {len(files)} H5 files."
        )

    elif (
        Path(input_path).is_file() and Path(input_path).suffix in ALLOWED_H5_EXTENSIONS
    ):
        files = [Path(input_path)]
        logger.info(f"Converting from file: {input_path}")

    Path(zarr_dir).mkdir(parents=True, exist_ok=True)
    image_list_updates = []
    for file in files:
        new_zarr_url = convert_single_h5_to_ome(
            zarr_dir=zarr_dir,
            input_path=str(file),
            image_key=image_key,
            image_layout=image_layout,
            label_key=label_key,
            new_image_key=new_image_key,
            new_label_key=new_label_key,
            custom_axis=custom_axis,
            ome_zarr_parameters=ome_zarr_parameters,
        )

        if VALID_IMAGE_LAYOUT(image_layout) in [
            VALID_IMAGE_LAYOUT.CYX,
            VALID_IMAGE_LAYOUT.YX,
        ]:
            is_3d = False
        else:
            is_3d = True

        logger.info(f"Succesfully converted {file} to {new_zarr_url}")
        image_update = {"zarr_url": new_zarr_url, "types": {"is_3D": is_3d}}
        image_list_updates.append(image_update)

    return {"image_list_updates": image_list_updates}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=convert_h5_to_ome_zarr,
        logger_name=logger.name,
    )
