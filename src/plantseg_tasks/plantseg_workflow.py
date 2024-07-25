"""PlantSeg Workflow as a Fractal Task."""

from typing import Optional

from pydantic import validate_call

from plantseg_tasks.ngio.ngff_image import NgffImage
from plantseg_tasks.task_utils.process import plantseg_standard_workflow
from plantseg_tasks.task_utils.ps_workflow_input_models import (
    PlantSegPredictionsModel,
    PlantSegSegmentationModel,
)


@validate_call
def plantseg_workflow(
    *,
    zarr_url: str,
    channel: int = 0,
    level: int = 0,
    table_name: Optional[str] = None,
    prediction_model: PlantSegPredictionsModel,
    segmentation_model: PlantSegSegmentationModel,
) -> None:
    """Full PlantSeg workflow.

    This function runs the full PlantSeg workflow on a OME-Zarr file.

    Args:
        zarr_url: The URL of the Zarr file.
        channel: The channel to use.
        level: The level to use.
        table_name: The name of the table.
        prediction_model: The prediction model.
        segmentation_model: The segmentation model.
    """
    ngff_image = NgffImage(zarr_url=zarr_url)
    image = ngff_image.get_multiscale_image(level=level)
    table_handler = ngff_image.get_roi_table(table_name=table_name)
    label = ngff_image.create_new_label("ps_test")
    label = label.change_level(level=level)

    max_seg_id = 0
    for info, patch in image.iter_over_rois(table_handler, return_info=True):
        if patch.ndim == 5:
            assert patch.shape[0] == 1, "Time dimension not supported"
            patch = patch[0]

        assert patch.ndim == 4, "Only 4D images are supported CXYZ"
        patch = patch[channel]

        seg = plantseg_standard_workflow(
            image=patch,
            prediction_model=prediction_model,
            segmentation_model=segmentation_model,
        )
        seg += max_seg_id
        max_seg_id = seg.max() + 1

        slices = info.slices[-3:]
        label._write_data(seg, slices=slices)

    label.consolidate()


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(task_function=plantseg_standard_workflow)
