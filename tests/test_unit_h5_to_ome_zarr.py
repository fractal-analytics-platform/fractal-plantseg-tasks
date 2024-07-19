from pathlib import Path

import numpy as np
import pytest
from plantseg.io import create_h5

from plantseg_tasks.task_utils.io import load_h5_images


@pytest.fixture
def sample_h5_file(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Create a sample h5 file for testing."""
    h5_file = tmp_path / "sample.h5"
    voxel_size = (0.5, 0.25, 0.25)
    random_image = np.random.randint(0, 255, (10, 10, 10))
    random_label = np.random.randint(0, 2, (10, 10, 10))
    create_h5(path=h5_file, stack=random_image, key="raw", voxel_size=voxel_size)
    create_h5(path=h5_file, stack=random_label, key="label", voxel_size=voxel_size)
    return h5_file, {"image_key": "raw", "label_key": "label"}


class TestH5ToOmeZarr:
    def test_load_h5_image_and_label(self, sample_h5_file: tuple[Path, dict[str, str]]):
        sample_h5_file, keys = sample_h5_file
        image_dc = load_h5_images(
            input_path=sample_h5_file,
            image_key=keys["image_key"],
            label_key=keys["label_key"],
            image_layout="ZYX",
        )
        assert image_dc.image_data.shape == (10, 10, 10)
        assert image_dc.label_data.shape == (10, 10, 10)
        np.testing.assert_allclose((0.5, 0.25, 0.25), image_dc.voxel_size)
        assert image_dc.unit == "um"

    def test_load_h5_image(self, sample_h5_file: tuple[Path, dict[str, str]]):
        sample_h5_file, keys = sample_h5_file
        image_dc = load_h5_images(
            input_path=sample_h5_file,
            image_key=keys["image_key"],
            label_key=None,
            image_layout="ZYX",
        )
        assert image_dc.image_data.shape == (10, 10, 10)
        assert image_dc.label_data is None
        np.testing.assert_allclose((0.5, 0.25, 0.25), image_dc.voxel_size)
        assert image_dc.unit == "um"

    def test_fail_if_key_not_found(self, sample_h5_file: tuple[Path, dict[str, str]]):
        sample_h5_file, keys = sample_h5_file
        with pytest.raises(KeyError):
            load_h5_images(
                input_path=sample_h5_file,
                image_key="not_found",
                label_key=keys["label_key"],
                image_layout="ZYX",
            )

    def test_fail_if_layout_not_correct(
        self, sample_h5_file: tuple[Path, dict[str, str]]
    ):
        sample_h5_file, keys = sample_h5_file
        with pytest.raises(ValueError):
            load_h5_images(
                input_path=sample_h5_file,
                image_key=keys["image_key"],
                label_key=keys["label_key"],
                image_layout="YX",
            )
