"""IO utils for Converters."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from plantseg.io import load_h5, load_tiff

from plantseg_tasks.task_utils.converter_input_models import (
    VALID_IMAGE_LAYOUT,
    CustomAxisInputModel,
    OMEZarrBuilderParams,
)


def create_ome_zarr(
    zarr_url: str,
    path: str,
    image_data: np.ndarray,
    voxel_size: tuple[float],
    unit: str,
    custom_axis: CustomAxisInputModel,
    omezarr_params: OMEZarrBuilderParams,
):
    """TODO: Implement this function."""
    pass


@dataclass
class Image:
    """Simple dataclass to store image data."""

    image_key: str
    image_data: np.ndarray
    label_key: Optional[str]
    label_data: Optional[np.ndarray]
    voxel_size: tuple[float]
    unit: str
    layout: VALID_IMAGE_LAYOUT
    channel_axis: Optional[int] = None

    def __post_init__(self):
        """Post init method to validate the image data."""
        self.layout = VALID_IMAGE_LAYOUT(self.layout)

        if self.layout not in VALID_IMAGE_LAYOUT:
            raise ValueError(f"Invalid image layout: {self.layout}")

        match self.layout:
            case "YX":
                if len(self.image_data.shape) != 2:
                    self._raise_shape_error("image")
                if self.label_data is not None:
                    if len(self.label_data.shape) != 2:
                        self._raise_shape_error("label")
                    if self.image_data.shape != self.label_data.shape:
                        self._raise_shape_mismatch_error()

            case "ZYX":
                if len(self.image_data.shape) != 3:
                    self._raise_shape_error("image")
                if self.label_data is not None:
                    if len(self.label_data.shape) != 3:
                        self._raise_shape_error("label")
                    if self.image_data.shape != self.label_data.shape:
                        self._raise_shape_mismatch_error()
            case "CYX":
                if len(self.image_data.shape) != 3:
                    self._raise_shape_error("image")
                if self.label_data is not None:
                    if len(self.label_data.shape) != 2:
                        self._raise_shape_error("label")
                    if self.image_data.shape[1:] != self.label_data.shape:
                        self._raise_shape_mismatch_error()

                self.channel_axis = 0
            case "ZCYX":
                if len(self.image_data.shape) != 4:
                    self._raise_shape_error("image")

                _image_spatial_shape = [
                    self.image_data.shape[0],
                    self.image_data.shape[2],
                    self.image_data.shape[3],
                ]
                if self.label_data is not None:
                    if len(self.label_data.shape) != 3:
                        self._raise_shape_error("label")
                    if _image_spatial_shape != self.label_data.shape:
                        self._raise_shape_mismatch_error()
                self.channel_axis = 1
            case "CZYX":
                if len(self.image_data.shape) != 4:
                    self._raise_shape_error("image")
                if self.label_data is not None:
                    if len(self.label_data.shape) != 3:
                        self._raise_shape_error("label")
                    if self.image_data.shape[1:] != self.label_data.shape:
                        self._raise_shape_mismatch_error()
                self.channel_axis = 0

    def has_valid_voxel_size(self):
        """Check if the voxel size is valid.

        Plantseg will set the voxel size to [1, 1, 1] if it is not provided.
        """
        if np.allclose(np.prod(self.voxel_size), 1):
            return False
        return True

    def _raise_shape_error(self, mode: str):
        if mode == "image":
            raise ValueError(
                f"Invalid image shape {self.image_data.shape} for layout {self.layout}"
            )
        elif mode == "label":
            raise ValueError(
                f"Invalid label shape {self.label_data.shape} for layout {self.layout}"
            )
        raise ValueError(f"Invalid shape for mode {mode}")

    def _raise_shape_mismatch_error(self):
        raise ValueError(
            "Image and label spatial dimension do not match: "
            f"{self.image_data.shape} != {self.label_data.shape}, layout: {self.layout}"
        )


def load_h5_images(
    input_path: str,
    image_key: str = "raw",
    label_key: Optional[str] = None,
    new_image_key: Optional[str] = None,
    new_label_key: Optional[str] = None,
    image_layout: VALID_IMAGE_LAYOUT = "ZYX",
) -> Image:
    """Load images from an H5 file.

    From a given H5 file, load the image and optionally label data.

    Args:
        input_path (str): Path to the H5 file.
        image_key (str): Key to the image data in the H5 file.
        label_key (Optional[str]): Key to the label data in the H5 file.
        new_image_key (Optional[str]): New key for the image data to
            be stored in the OME-Zarr.
        new_label_key (Optional[str]): New key for the label data to
            be stored in the OME-Zarr.
        image_layout (VALID_IMAGE_LAYOUT): The layout of the image data.
    """
    if Path(input_path).suffix != ".h5":
        raise ValueError("plantseg expects only H5 files.")

    if not Path(input_path).exists():
        raise FileNotFoundError(f"File {input_path} not found.")

    image, (voxel_size, _, _, unit) = load_h5(input_path, key=image_key)

    if label_key is not None:
        label, _ = load_h5(input_path, key=label_key)
    else:
        label = None

    image_key = image_key if new_image_key is None else new_image_key
    label_key = label_key if new_label_key is None else new_label_key

    return Image(
        image_key=image_key,
        image_data=image,
        label_key=label_key,
        label_data=label,
        voxel_size=voxel_size,
        unit=unit,
        layout=image_layout,
    )


def load_tiff_images(
    image_path: str,
    label_path: Optional[str] = None,
    new_image_key: str = "raw",
    new_label_key: str = "label",
    image_layout: VALID_IMAGE_LAYOUT = "ZYX",
):
    """Load images from a TIFF files.

    From a given TIFF file, load the image and optionally
    label data (from a second TIFF file).

    Args:
        image_path (str): Path to the TIFF file.
        label_path (Optional[str]): Path to the label TIFF file.
        new_image_key (str): New key for the image data to be stored in the OME-Zarr.
        new_label_key (str): New key for the label data to be stored in the OME-Zarr.
        image_layout (VALID_IMAGE_LAYOUT): The layout of the image data.
    """
    image, (voxel_size, _, _, unit) = load_tiff(image_path)

    if label_path is not None:
        label, _ = load_tiff(label_path)
    else:
        label = None

    return Image(
        image_key=new_image_key,
        image_data=image,
        label_key=new_label_key,
        label_data=label,
        voxel_size=voxel_size,
        unit=unit,
        layout=image_layout,
    )
