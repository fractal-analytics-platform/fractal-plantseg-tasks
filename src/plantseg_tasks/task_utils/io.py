"""IO utils for Converters."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import zarr
import zarr.storage
from fractal_tasks_core import __OME_NGFF_VERSION__
from fractal_tasks_core.ngff.specs import (
    Axis,
    Dataset,
    Multiscale,
    NgffImageMeta,
    ScaleCoordinateTransformation,
)
from fractal_tasks_core.pyramids import build_pyramid
from plantseg.io import load_h5, load_tiff

from plantseg_tasks.task_utils.converter_input_models import (
    VALID_IMAGE_LAYOUT,
    CustomAxisInputModel,
    OMEZarrBuilderParams,
)


def to_standard_layout(
    image_data: np.ndarray,
    current_layout: VALID_IMAGE_LAYOUT,
    voxel_size: tuple = (1, 1, 1),
    standard_layout="CZYX",
):
    """Convert any layout to standard layout."""
    layout_as_str = current_layout.value

    if len(layout_as_str) != image_data.ndim:
        raise ValueError(
            f"Image data has {image_data.ndim} dimensions and shape {image_data.shape},"
            f" but the current layout is {len(layout_as_str)}"
            f" dimensions and shape {layout_as_str}. Please provide a different layout."
        )

    axis_shape = {ax: 1 for ax in standard_layout}
    axis_pos = {ax: None for i, ax in enumerate(standard_layout)}

    for index, ax in enumerate(layout_as_str):
        axis_shape[ax] = image_data.shape[index]
        axis_pos[ax] = index

    transpose_order = tuple(
        [axis_pos[ax] for ax in standard_layout if axis_pos[ax] is not None]
    )
    if transpose_order != tuple(range(image_data.ndim)):
        image_data = np.transpose(image_data, transpose_order)

    image_data = image_data.reshape(tuple(axis_shape.values()))

    if len(standard_layout) == 3:
        scale = voxel_size
    elif len(standard_layout) == 4:
        scale = [1, *voxel_size]
    else:
        raise ValueError("Invalid number of dimensions.")

    return image_data, scale


@dataclass
class Label:
    """Simple dataclass to store label data."""

    label_key: str
    label_data: np.ndarray
    voxel_size: tuple[float] = (1, 1, 1)
    unit: str = "micrometer"
    layout: VALID_IMAGE_LAYOUT = "ZYX"
    _scale = (1, 1, 1, 1)

    def __post_init__(self):
        """Post init method to validate the label data."""
        if len(self.voxel_size) != 3:
            raise ValueError("Voxel size must be a 3-tuple.")

        label, scale = to_standard_layout(
            image_data=self.label_data,
            current_layout=VALID_IMAGE_LAYOUT(self.layout),
            voxel_size=self.voxel_size,
            standard_layout="ZYX",
        )

        self.label_data = label
        self._scale = scale
        self.layout = "ZYX"
        self._units = [self.unit, self.unit, self.unit]
        self._axis_type = ["space", "space", "space"]

    @property
    def axis_info(self):
        """Return the axis information."""
        return zip(self.layout.lower(), self._units, self._axis_type)

    @property
    def scale(self) -> tuple[float]:
        """Return the scale of the label."""
        return self._scale


@dataclass
class Image:
    """Simple dataclass to store image data.

    By default the image layout is going to be internaly converted to "TCZYX" layout.
    """

    image_key: str
    image_data: np.ndarray
    voxel_size: tuple[float]
    unit: str
    layout: VALID_IMAGE_LAYOUT
    channel_names: Optional[list[str]] = None
    label: Optional[Label] = None
    type: str = "image"
    _scale: tuple[float] = (1, 1, 1, 1)
    input_layout: Optional[VALID_IMAGE_LAYOUT] = None

    def __post_init__(self):
        """Post init method to validate the image data."""
        layout = VALID_IMAGE_LAYOUT(self.layout)
        self.input_layout = layout

        if self.type == "image":
            standard_layout = "CZYX"
        elif self.type == "label":
            standard_layout = "ZYX"
        else:
            raise ValueError()

        if layout not in VALID_IMAGE_LAYOUT:
            raise ValueError(f"Invalid image layout: {self.layout}")

        if len(self.voxel_size) != 3:
            raise ValueError("Voxel size must be a 3-tuple.")

        image, scale = to_standard_layout(
            image_data=self.image_data,
            current_layout=layout,
            voxel_size=self.voxel_size,
            standard_layout=standard_layout,
        )

        self.image_data = image
        self._scale = scale
        self.layout = standard_layout
        self._axis_units = [None, self.unit, self.unit, self.unit]
        self._axis_type = ["channel", "space", "space", "space"]

        if self.label is not None:
            self.label = Label(
                label_key=self.label.label_key,
                label_data=self.label.label_data,
                layout=self.input_layout,
                voxel_size=self.voxel_size,
                unit=self.unit,
            )

    @property
    def axis_info(self):
        """Return the axis information."""
        return zip(self.layout.lower(), self._axis_units, self._axis_type)

    @property
    def scale(self) -> tuple[float]:
        """Return the scale of the image."""
        return self._scale

    def has_valid_voxel_size(self):
        """Check if the voxel size is valid.

        Plantseg will set the voxel size to [1, 1, 1] if it is not provided.
        """
        if np.allclose(np.prod(self.voxel_size), 1):
            return False
        return True


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
    image_key = image_key if new_image_key is None else new_image_key

    if label_key is not None:
        label_key = label_key if new_label_key is None else new_label_key
        _label, _ = load_h5(input_path, key=label_key)
        label = Label(
            label_key=label_key,
            label_data=_label,
            voxel_size=voxel_size,
            unit=unit,
            layout=image_layout,
        )
    else:
        label = None

    return Image(
        image_key=image_key,
        image_data=image,
        voxel_size=voxel_size,
        unit=unit,
        layout=image_layout,
        label=label,
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
    _image, (voxel_size, _, _, unit) = load_tiff(image_path)

    if label_path is not None:
        _label, _ = load_tiff(label_path)
        label = Label(
            label_key=new_label_key,
            label_data=_label,
            voxel_size=voxel_size,
            unit=unit,
            layout=image_layout,
        )
    else:
        label = None

    return Image(
        image_key=new_image_key,
        image_data=_image,
        voxel_size=voxel_size,
        unit=unit,
        layout=image_layout,
        label=label,
    )


def correct_image_metadata(image: Image, custom_axis: CustomAxisInputModel) -> Image:
    """If the image does not have a valid voxel size, set it from the custom axis.

    TODO: make sure that a custom axis is provided before returning the image.
    """
    if image.has_valid_voxel_size():
        return image

    voxel_size = [1, None, None]
    for ax in custom_axis.axis:
        if ax.axis_name == "z":
            voxel_size[0] = ax.scale
        if ax.axis_name == "y":
            voxel_size[1] = ax.scale
        if ax.axis_name == "x":
            voxel_size[2] = ax.scale

    if None in voxel_size:
        raise ValueError(
            "No valid voxel size found in the file or custom axis."
            "Please provide a valid voxel size for axis X and Y."
        )

    label = Label(
        label_key=image.label.label_key,
        label_data=image.label.label_data,
        voxel_size=voxel_size,
        unit=custom_axis.unit,
        layout=image.input_layout,
    )

    return Image(
        image_key=image.image_key,
        image_data=image.image_data,
        voxel_size=voxel_size,
        unit=custom_axis.spatial_units,
        layout=image.layout,
        channel_names=image.channel_names,
        label=label,
    )


def build_multiscale_metadata(
    image: Image,
    omezarr_params: OMEZarrBuilderParams,
    name: str,
) -> Multiscale:
    """Build the multiscale metadata for the OME-Zarr file."""
    axes_metadata = [Axis(name=n, unit=u, type=t) for n, u, t in image.axis_info]

    datasets = []
    if omezarr_params.scaling_factor_Z != 1:
        raise NotImplementedError("Z scaling is not yet supported.")
    downsample_factors = [
        1,
        omezarr_params.scaling_factor_Z,
        int(omezarr_params.scaling_factor_XY),
        int(omezarr_params.scaling_factor_XY),
    ]
    scale = image.scale
    for i in range(omezarr_params.number_multiscale):
        _dataset = Dataset(
            path=str(i),
            coordinateTransformations=[
                ScaleCoordinateTransformation(type="scale", scale=scale)
            ],
        )
        datasets.append(_dataset)
        scale = [s * f for s, f in zip(scale, downsample_factors)]

    return Multiscale(
        version=__OME_NGFF_VERSION__,
        name=name,
        axes=axes_metadata,
        datasets=datasets,
    )


def create_ome_zarr(
    zarr_url: str,
    path: str,
    name: str,
    image: Image,
    omezarr_params: OMEZarrBuilderParams,
) -> str:
    """Create an OME-Zarr file from a give Image object."""
    multiscale_metadata = build_multiscale_metadata(
        image=image, omezarr_params=omezarr_params, name=name
    )
    omero_metadata = None

    ngff_metadata = NgffImageMeta(
        multiscales=[multiscale_metadata],
        omero=omero_metadata,
    )

    chunksize = (1, 1, image.image_data.shape[-2], image.image_data.shape[-1])
    image_data = image.image_data
    shape = image_data.shape
    zarr_url = f"{zarr_url}/{path}"
    zarr_group = zarr.open_group(store=zarr.storage.FSStore(f"{zarr_url}"), mode="w")
    zarr_group.attrs.update(ngff_metadata.model_dump(exclude_none=True))
    zarr_array = zarr.open_array(
        shape=shape,
        chunks=chunksize,
        dtype=image.image_data.dtype,
        store=zarr.storage.FSStore(f"{zarr_url}/0"),
        overwrite=True,
        dimension_separator="/",
    )
    zarr_array[...] = image_data
    build_pyramid(
        zarrurl=zarr_url,
        num_levels=omezarr_params.number_multiscale,
        coarsening_xy=int(omezarr_params.scaling_factor_XY),
    )

    if image.label is not None:
        # Create the label group container
        label_container_group = zarr.open_group(
            store=zarr.storage.FSStore(f"{zarr_url}/labels"), mode="w"
        )
        label_container_group.attrs["labels"] = [image.label.label_key]

        # Prepare the label metadata
        label_url_path = f"{zarr_url}/labels/{image.label.label_key}"

        label_group = zarr.open_group(
            store=zarr.storage.FSStore(label_url_path), mode="w"
        )
        label_source = {
            "image-label": {
                "version": __OME_NGFF_VERSION__,
                "source": {"image": "../../"},
            }
        }
        label_group.attrs.update(label_source)

        label_metadata = build_multiscale_metadata(
            image=image.label, omezarr_params=omezarr_params, name="label"
        )
        label_ngff_metadata = NgffImageMeta(
            multiscales=[label_metadata],
            omero=None,
        )

        label_group.attrs.update(label_ngff_metadata.model_dump(exclude_none=True))

        # Write the label data
        label_zarr_array = zarr.open_array(
            shape=image.label.label_data.shape,
            chunks=chunksize[1:],
            dtype=image.label.label_data.dtype,
            store=zarr.storage.FSStore(label_url_path + "/0"),
            overwrite=True,
            dimension_separator="/",
        )
        label_zarr_array[...] = image.label.label_data

        build_pyramid(
            zarrurl=label_url_path,
            num_levels=omezarr_params.number_multiscale,
            coarsening_xy=int(omezarr_params.scaling_factor_XY),
        )
    return zarr_url
