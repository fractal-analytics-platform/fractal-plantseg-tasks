"""This task converts simple H5 files to OME-Zarr."""

from fractal_tasks_core.utils import logger
from pydantic import validate_call

from plantseg_tasks.task_utils.ps_workflow_input_models import (
    PlantSegPredictionsModel,
    PlantSegSegmentationModel,
)

from fractal_tasks_core.tasks.io_models import ChannelInputModel


@validate_call
def plantseg_workflow(
    zarr_url: str,
    channel_model: ChannelInputModel,
    prediction_param: PlantSegPredictionsModel,
    segmentation_param: PlantSegSegmentationModel,
):
    """PlantSeg workflow task.

    Args:
        zarr_url: Zarr url
        channel_model (ChannelInputModel):
        predictions_params (PlantSegPredictionsModel): Configuration for
            PlantSeg predictions step.
        segmentation_params (PlantSegSegmentationModel): Configuration for
            PlantSeg segmentation step.
    """
    """
        #predictions_params (PlantSegPredictionsModel): Configuration for
            PlantSeg predictions step.
        segmentation_params (PlantSegSegmentationModel): Configuration for
            PlantSeg segmentation step.
    """
    pass


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=plantseg_workflow,
        logger_name=logger.name,
    )
