"""This module provides a class abstraction for a OME-NGFF image."""

import zarr
from fractal_tasks_core.ngff.specs import (
    Dataset,
    Multiscale,
    NgffImageMeta,
    Omero,
    ScaleCoordinateTransformation,
)

from plantseg_tasks.ngio.multiscale_handlers import MultiscaleImage, MultiscaleLabel
from plantseg_tasks.ngio.table_handlers import RoiTableHandler


class NgffImage:
    """A class to handle OME-NGFF images."""

    def __init__(self, zarr_url: str, mode: str = "r") -> None:
        """Initialize the NGFFImage in read mode."""
        # setup the main image
        self._zarr_url = zarr_url
        self.group = zarr.open_group(zarr_url, mode=mode)

    @property
    def list_labels(self) -> list[str]:
        """List all the labels in the image."""
        labels_group = self.group.get("labels", None)
        if labels_group is None:
            return []

        list_labels = labels_group.attrs["labels"]
        if not isinstance(list_labels, list):
            raise ValueError("Labels must be a list of strings.")
        return list_labels

    @property
    def list_tables(self) -> list[str]:
        """List all the tables in the image."""
        tables_group = self.group.get("tables", None)
        if tables_group is None:
            return []

        list_tables = tables_group.attrs["tables"]
        if not isinstance(list_tables, list):
            raise ValueError("Tables must be a list of strings.")
        return list_tables

    @property
    def zarr_url(self) -> str:
        """Return the Zarr URL of the image."""
        return self._zarr_url

    def get_multiscale_label(self, label_name: str, level: int = 0) -> MultiscaleLabel:
        """Create a MultiscaleLabel object."""
        return MultiscaleLabel(self.zarr_url, path=f"labels/{label_name}", level=level)

    def get_roi_table(self, table_name: str) -> RoiTableHandler:
        """Create a RoiTableHandler object."""
        return RoiTableHandler(self.zarr_url, table_name)

    def get_multiscale_image(self, level: int = 0) -> MultiscaleImage:
        """Create a MultiscaleImage object."""
        return MultiscaleImage(zarr_url=self.zarr_url, level=level)

    def derive_new_ngff_image(
        self,
        zarr_url: str,
        channel_names: list[str] | None = None,
        shape: tuple[int, ...] | None = None,
        copy_tables: bool | list[str] = False,
        copy_labels: bool | list[str] = False,
    ) -> "NgffImage":
        """Derive a new image from the current image."""
        from plantseg_tasks.ngio.ngff.ngff_writers import (
            create_ngff_image_from_metadata,
        )

        multiscale_image = self.get_multiscale_image()

        if shape is None:
            shape = multiscale_image.shape

        multiscale_image_metadata = multiscale_image.metadata
        multiscale_image_metadata.omero = None
        # TODO andle type more carefully
        return create_ngff_image_from_metadata(
            zarr_url=zarr_url,
            shape=shape,
            dtype="uint16",
            metadata=multiscale_image_metadata,
        )

    def create_new_label(
        self,
        new_label_name: str,
    ) -> "MultiscaleLabel":
        """Create a new label in the current image."""
        label_group = zarr.open(self.zarr_url, mode="a").require_group("labels")

        if "labels" not in label_group.attrs:
            label_group.attrs["labels"] = []

        labels = label_group.attrs["labels"]
        label_group.attrs["labels"] = labels + [new_label_name]

        label_group = zarr.open(self.zarr_url, path="labels", mode="a").require_group(
            new_label_name
        )

        shape = self.get_multiscale_image().shape
        new_shape = (shape[1], shape[2], shape[3])

        image = self.get_multiscale_image()
        for i in image.list_levels:
            image = image.change_level(i)
            shape = image.shape
            new_shape = (shape[1], shape[2], shape[3])
            new_label_dataset = zarr.open_array(
                f"{self.zarr_url}/labels/{new_label_name}/{i}",
                shape=new_shape,
                dtype="<i4",
                mode="w",
                dimension_separator="/",
            )

        multiscale = self.get_multiscale_image().metadata.multiscales[0]
        multiscale_axes = multiscale.axes[1:]
        new_dataset = []
        for dataset in multiscale.datasets:
            scale = dataset.coordinateTransformations[0].scale[1:]
            new_dataset.append(
                Dataset(
                    path=dataset.path,
                    coordinateTransformations=[
                        ScaleCoordinateTransformation(type="scale", scale=scale)
                    ],
                )
            )

        metadata = NgffImageMeta(
            multiscales=[
                Multiscale(
                    name=new_label_name,
                    version="0.4",
                    axes=multiscale_axes,
                    datasets=new_dataset,
                )
            ]
        )
        label_group.attrs.update(metadata.dict(exclude_none=True))
        source_dict = {
            "image-label": {"source": {"image": "../../"}, "version": "0.4"},
        }
        label_group.attrs.update(source_dict)
        return self.get_multiscale_label(new_label_name)

    def derive_new_label(
        self,
        new_label_name: str,
        source_label_name: str,
    ) -> "MultiscaleLabel":
        """Derive a new label from a current label."""
        raise NotImplementedError

    def create_new_table(self, new_table_name: str) -> RoiTableHandler:
        """Create a new table in the current image."""
        raise NotImplementedError
