"""This task converts simple H5 files to OME-Zarr."""

from pathlib import Path
from typing import Optional

from fractal_tasks_core.utils import logger
from pydantic import Field, validate_call

from plantseg_tasks.task_utils.converter_input_models import (
    ALLOWED_TIFF_EXTENSIONS,
    VALID_IMAGE_LAYOUT,
    CustomAxisInputModel,
    OMEZarrBuilderParams,
)
from plantseg_tasks.task_utils.io import (
    correct_image_metadata,
    create_ome_zarr,
    load_tiff_images,
)


def convert_single_tiff_to_ome_zarr(
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
) -> str:
    """TIFF to OME-Zarr converter task.

    Args:
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
    image_ds = load_tiff_images(
        image_path=image_path,
        label_path=label_path,
        new_image_key=new_image_key,
        new_label_key=new_label_key,
        image_layout=image_layout,
    )

    zarr_url = Path(zarr_dir) / f"{Path(image_path).stem}.zarr"
    image_ds = correct_image_metadata(image_ds, custom_axis=custom_axis)
    return create_ome_zarr(
        zarr_url=zarr_url,
        path=image_ds.image_key,
        name=image_ds.image_key,
        image=image_ds,
        omezarr_params=ome_zarr_parameters,
    )


@validate_call
def convert_tiff_to_ome_zarr(
    zarr_urls: list[str],
    zarr_dir: str,
    image_path: str,
    image_layout: VALID_IMAGE_LAYOUT = "ZYX",
    label_path: Optional[str] = None,
    new_image_key: str = "raw",
    new_label_key: str = "label",
    custom_axis: CustomAxisInputModel = Field(
        title="Custom Axis", default=CustomAxisInputModel()
    ),
    ome_zarr_parameters: OMEZarrBuilderParams = Field(
        title="OME-Zarr Parameters", default=OMEZarrBuilderParams()
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
    if not Path(image_path).exists():
        raise ValueError(f"Input path {image_path} does not exist.")

    if label_path is not None and not Path(label_path).exists():
        raise ValueError(f"Label path {label_path} does not exist.")

    if Path(image_path).is_dir() and label_path is None:
        files = []
        for ext in ALLOWED_TIFF_EXTENSIONS:
            files += list(Path(image_path).glob(f"*{ext}"))
    elif Path(image_path).is_dir() and label_path is not None:
        raise NotImplementedError(
            "Label path must be None if image path is a folder. "
            "Batch conversion is not yet supported."
        )
    elif (
        Path(image_path).is_file()
        and Path(image_path).suffix in ALLOWED_TIFF_EXTENSIONS
    ):
        files = [Path(image_path)]

    image_list_updates = []
    for file in files:
        new_zarr_url = convert_single_tiff_to_ome_zarr(
            zarr_dir=zarr_dir,
            image_path=str(file),
            image_layout=image_layout,
            label_path=label_path,
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

        image_update = {"zarr_url": new_zarr_url, "types": {"is_3D": is_3d}}
        image_list_updates.append(image_update)

    return {"image_list_updates": image_list_updates}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=convert_tiff_to_ome_zarr,
        logger_name=logger.name,
    )
