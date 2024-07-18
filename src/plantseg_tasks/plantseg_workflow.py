"""This task converts simple H5 files to OME-Zarr."""

from fractal_tasks_core.utils import logger
from pydantic.v1.decorator import validate_arguments

from plantseg_tasks.task_utils.ps_workflow_input_models import (
    PlantSegPredictionsModel,
    PlantSegSegmentationModel,
)


@validate_arguments
def plantseg_workflow(
    predictions_params: PlantSegPredictionsModel,
    segmentation_params: PlantSegSegmentationModel,
):
    """PlantSeg workflow task.

    Args:
        predictions_params (PlantSegPredictionsModel): Configuration for
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
