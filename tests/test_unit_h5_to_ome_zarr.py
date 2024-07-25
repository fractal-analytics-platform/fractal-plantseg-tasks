from pathlib import Path

import numpy as np
import pytest
from fractal_tasks_core.ngff.zarr_utils import load_NgffImageMeta
from plantseg.io import create_h5

from plantseg_tasks.convert_h5_to_ome_zarr import convert_h5_to_ome_zarr


@pytest.fixture
def sample_h5_file_3d(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Create a sample h5 file for testing."""
    h5_file = tmp_path / "sample.h5"
    voxel_size = (0.5, 0.25, 0.25)
    random_image = np.random.randint(0, 255, (10, 10, 10))
    random_label = np.random.randint(0, 2, (10, 10, 10))
    create_h5(path=h5_file, stack=random_image, key="raw", voxel_size=voxel_size)
    create_h5(path=h5_file, stack=random_label, key="label", voxel_size=voxel_size)
    return h5_file, {"image_key": "raw", "label_key": "label"}


class TestH5ToOmeZarr:
    def test_full_workflow_3D(self, sample_h5_file_3d: tuple[str, str], tmp_path: Path):
        zarr_dir = str(tmp_path / "zarr")
        image_path, keys = sample_h5_file_3d
        image_path = str(image_path)

        # Happy path
        image_list_update = convert_h5_to_ome_zarr(
            zarr_urls=[],
            zarr_dir=zarr_dir,
            input_path=image_path,
            image_key=keys["image_key"],
            label_key=keys["label_key"],
            image_layout="ZYX",
        )

        zarr_url = image_list_update["image_list_updates"][0]["zarr_url"]
        assert Path(zarr_url).exists()
        load_NgffImageMeta(zarr_url)

        # TODO add proper validation with ngio
