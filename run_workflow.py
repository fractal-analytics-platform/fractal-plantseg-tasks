from src.plantseg_tasks.plantseg_workflow import plantseg_workflow

"""PlantSeg Workflow as a Fractal Task."""

from typing import Optional

from pydantic import validate_call

from plantseg_tasks.ngio.ngff_image import NgffImage
from plantseg_tasks.task_utils.process import plantseg_standard_workflow
from plantseg_tasks.task_utils.ps_workflow_input_models import (
    PlantSegPredictionsModel,
    PlantSegSegmentationModel,
)

"""

@validate_call
def plantseg_workflow(
    *,
    zarr_url: str,
    channel: int = 0,
    level: int = 0,
    table_name: Optional[str] = None,
    prediction_model: PlantSegPredictionsModel,
    segmentation_model: PlantSegSegmentationModel,
    label_name: Optional[str] = None,

"""


prediction_model = PlantSegPredictionsModel(
    model_source="PlantSegZoo",
    plantsegzoo_name="generic_confocal_3D_unet",
    device="cuda",
    skip=False,
)
segmentation_model = PlantSegSegmentationModel()
plantseg_workflow(
    zarr_url="/home/lcerrone/data/ome-zarr/ovule_sample.zarr/raw",
    channel=0,
    level=0,
    prediction_model=prediction_model,
    segmentation_model=segmentation_model,
)
