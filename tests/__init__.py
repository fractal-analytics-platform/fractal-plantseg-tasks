import json
from pathlib import Path

import plantseg_tasks
from plantseg.io import create_h5, create_tiff
import numpy as np

PACKAGE = "plantseg_tasks"
PACKAGE_DIR = Path(plantseg_tasks.__file__).parent
MANIFEST_FILE = PACKAGE_DIR / "__FRACTAL_MANIFEST__.json"
with MANIFEST_FILE.open("r") as f:
    MANIFEST = json.load(f)
    TASK_LIST = MANIFEST["task_list"]


"""
@fixture
def sample_tiff_file(tmp_path: Path) -> tuple[Path, Path]:
    tiff_file = tmp_path / "sample.tiff"
    tiff_label_file = tmp_path / "sample_label.tiff"
    voxel_size = (0.5, 0.25, 0.25)
    random_image = np.random.randint(0, 255, (10, 10, 10))
    random_label = np.random.randint(0, 2, (10, 10, 10))
    create_tiff(
        tiff_file, stack=random_image, voxel_size=voxel_size, voxel_size_unit="um"
    )
    create_tiff(
        tiff_label_file, stack=random_label, voxel_size=voxel_size, voxel_size_unit="um"
    )
    return tiff_file, tiff_label_file
"""
