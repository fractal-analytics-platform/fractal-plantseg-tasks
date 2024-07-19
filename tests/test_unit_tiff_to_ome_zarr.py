from pathlib import Path

import numpy as np
import pytest
from plantseg.io import create_tiff

from plantseg_tasks.task_utils.io import load_tiff_images


@pytest.fixture
def sample_tiff_file(tmp_path: Path) -> tuple[Path, Path]:
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
    return tiff_file, tiff_label_file


class TestTiffToOmeZarr:
    def test_load_tiff_image_and_label(self, sample_tiff_file: tuple[Path, Path]):
        image_file, label_file = sample_tiff_file
        image_dc = load_tiff_images(
            image_path=image_file, label_path=label_file, image_layout="ZYX"
        )
        assert image_dc.image_data.shape == (10, 10, 10)
        assert image_dc.label_data.shape == (10, 10, 10)
        np.testing.assert_allclose((0.5, 0.25, 0.25), image_dc.voxel_size)
        assert image_dc.unit == "um"

    def test_load_tiff_image(self, sample_tiff_file: tuple[Path, Path]):
        image_file, _ = sample_tiff_file
        image_dc = load_tiff_images(
            image_path=image_file, label_path=None, image_layout="ZYX"
        )
        assert image_dc.image_data.shape == (10, 10, 10)
        assert image_dc.label_data is None
        np.testing.assert_allclose((0.5, 0.25, 0.25), image_dc.voxel_size)
        assert image_dc.unit == "um"

    def test_fail_if_file_not_found(self, sample_tiff_file: tuple[Path, Path]):
        image_file, _ = sample_tiff_file
        with pytest.raises(FileNotFoundError):
            load_tiff_images(
                image_path=image_file.with_suffix(".not_found"),
                label_path=None,
                image_layout="ZYX",
            )

    def test_fail_if_layout_not_correct(self, sample_tiff_file: tuple[Path, Path]):
        image_file, _ = sample_tiff_file
        with pytest.raises(ValueError):
            load_tiff_images(image_path=image_file, label_path=None, image_layout="YX")
