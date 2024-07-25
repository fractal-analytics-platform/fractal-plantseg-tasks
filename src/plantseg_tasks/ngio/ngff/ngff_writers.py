from pathlib import Path

import zarr
from fractal_tasks_core.ngff.specs import (
    Axis,
    Channel,
    Dataset,
    Multiscale,
    NgffImageMeta,
    Omero,
    ScaleCoordinateTransformation,
)

from plantseg_tasks.ngio.ngff_image import NgffImage


def create_ngff_image_from_metadata(
    zarr_url, shape, dtype, metadata: NgffImageMeta
) -> NgffImage:
    # this is a placeholder for the actual implementation
    new_ome_zarr = zarr.open(zarr_url, mode="w")
    for i, _ in enumerate(metadata.multiscales[0].datasets):
        new_ome_zarr.create_dataset(
            f"{i}", shape=shape, dimension_separator="/", dtype=dtype
        )
        shape = [shape[0], shape[1], shape[2] // 2, shape[3] // 2]

    new_ome_zarr.attrs.update(metadata.dict(exclude_none=True))
    return NgffImage(zarr_url)


def create_ngff_image(
    zarr_url: str | Path,
    shape: tuple[int, ...],
    pixel_resulution: tuple[float, ...],
    path: str | None = None,
    name: str | None = None,
    dtype: str | None = "uint16",
    chunks: str | None = None,
    spatial_units: str = "micrometer",
    channel_names: list[str] | None = None,
    axis_order: list[str] | None = None,
    scling_factor_xy: float = 2.0,
    num_levels: int = 3,
) -> NgffImage:
    if axis_order is None:
        axis_order = ["t", "c", "z", "y", "x"][-len(shape) :]

    list_axes = []
    for ax in axis_order:
        assert ax in ["x", "y", "z", "c", "t"]
        if ax in ["x", "y", "z"]:
            list_axes.append(Axis(name=ax, type="space", unit=spatial_units))
        elif ax == "c":
            list_axes.append(Axis(name=ax, type="channel"))
        elif ax == "t":
            list_axes.append(Axis(name=ax, type="time"))

    spatial_axis = [ax for ax in axis_order if ax in ["x", "y", "z"]]

    assert len(spatial_axis) == len(pixel_resulution)
    list_datasets = []
    list_shapes = []
    list_chunks = []

    for i in range(num_levels):
        list_datasets.append(
            Dataset(
                path=str(i),
                coordinateTransformations=[
                    ScaleCoordinateTransformation(type="scale", scale=pixel_resulution)
                ],
            )
        )
        pixel_resulution = [
            s * scling_factor_xy if ax in ["x", "y"] else s
            for s, ax in zip(pixel_resulution, spatial_axis)
        ]

        list_shapes.append(shape)
        shape = [
            int(s / scling_factor_xy) if ax in ["x", "y"] else s
            for s, ax in zip(shape, spatial_axis)
        ]
        if chunks is not None:
            list_chunks.append(chunks)
            channel_names = [
                int(c / scling_factor_xy) if ax in ["x", "y"] else c
                for c, ax in zip(chunks, spatial_axis)
            ]
        else:
            list_chunks.append(None)

    multiscale = Multiscale(
        name=name, datasets=list_datasets, axes=list_axes, version="0.4"
    )
    if channel_names is not None:
        list_channels = []
        for name in channel_names:
            list_channels.append(Channel(label=name, color="#ffffff"))
        omero_meta = Omero(channels=list_channels)
    else:
        omero_meta = None

    ngff_meta = NgffImageMeta(multiscales=[multiscale], omero=omero_meta)
    new_ome_zarr = zarr.open(zarr_url, mode="w")
    for (i, shape), chunks in zip(enumerate(list_shapes), list_chunks):
        new_ome_zarr.create_dataset(
            f"{i}", shape=shape, dtype=dtype, chunks=chunks, dimension_separator="/"
        )
    new_ome_zarr.attrs.update(ngff_meta.dict())

    return NgffImage(zarr_url)
