from pathlib import Path

import numpy as np
import pytest
from plantseg.io import create_tiff
from devtools import debug

from plantseg_tasks.convert_tiff_to_ome_zarr import convert_tiff_to_ome_zarr
from fractal_tasks_core.ngff.zarr_utils import load_NgffImageMeta


@pytest.fixture
def sample_tiff_file_3d(tmp_path: Path) -> tuple[str, str]:
    tiff_file = tmp_path / "sample.tiff"
    tiff_label_file = tmp_path / "sample_label.tiff"
    voxel_size = (0.5, 0.25, 0.25)
    random_image = np.random.randint(0, 255, (10, 10, 10)).astype("uint8")
    random_label = np.random.randint(0, 2, (10, 10, 10)).astype("uint16")
    create_tiff(
        tiff_file, stack=random_image, voxel_size=voxel_size, voxel_size_unit="um"
    )
    create_tiff(
        tiff_label_file, stack=random_label, voxel_size=voxel_size, voxel_size_unit="um"
    )
    return str(tiff_file), str(tiff_label_file)


class TestTiffToOmeZarr:
    def test_full_workflow_3D(
        self, sample_tiff_file_3d: tuple[str, str], tmp_path: Path
    ):
        zarr_dir = str(tmp_path / "zarr")
        image_path, label_path = sample_tiff_file_3d

        # Happy path
        image_list_update = convert_tiff_to_ome_zarr(
            zarr_urls=[],
            zarr_dir=zarr_dir,
            image_path=image_path,
            label_path=label_path,
            new_image_key="raw",
            new_label_key="label",
            image_layout="ZYX",
        )

        zarr_url = image_list_update["image_list_updates"][0]["zarr_url"]
        assert Path(zarr_url).exists()
        load_NgffImageMeta(zarr_url)

        # TODO add proper validation with ngio
