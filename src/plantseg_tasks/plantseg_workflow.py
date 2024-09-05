"""PlantSeg Workflow as a Fractal Task."""

from typing import Any, Optional

from fractal_tasks_core.utils import logger
from pydantic import validate_call

from plantseg_tasks.ngio.ngff_image import NgffImage
from plantseg_tasks.task_utils.process import plantseg_standard_workflow
from plantseg_tasks.task_utils.ps_workflow_input_models import (
    PlantSegPredictionsModel,
    PlantSegSegmentationModel,
)


def _predict_simple(image, label, channel, prediction_model, segmentation_model):
    logger.info("Predicting on the full image")
    patch = image.get_data()
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

    label.write_data(seg)
    return label


def _predict_with_roi(
    ngff_image, image, label, channel, prediction_model, segmentation_model, table_name
):
    logger.info(f"Predicting on ROIs from table {table_name}")
    table_handler = ngff_image.get_roi_table(table_name=table_name)

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
    return label


@validate_call
def plantseg_workflow(
    *,
    zarr_url: str,
    channel: int = 0,
    level: int = 0,
    table_name: Optional[str] = None,
    prediction_model: PlantSegPredictionsModel = PlantSegPredictionsModel(),
    segmentation_model: PlantSegSegmentationModel = PlantSegSegmentationModel(),
    label_name: Optional[str] = None,
) -> dict[str, Any]:
    """Full PlantSeg workflow.

    This function runs the full PlantSeg workflow on a OME-Zarr file.

    Args:
        zarr_url: The URL of the Zarr file.
        channel: Select the input channel to use.
        level: Select at which pyramid level to run the workflow.
        table_name: The name of a roi table to use.
        prediction_model: Parameters for the prediction model.
        segmentation_model: Parameters for the segmentation model.
        label_name: The name of the label to create with the plantseg segmentation.
    """
    ngff_image = NgffImage(zarr_url=zarr_url)
    image = ngff_image.get_multiscale_image(level=level)

    if label_name is None:
        label_name = f"plantseg_{segmentation_model.segmentation_type}"

    label = ngff_image.create_new_label(label_name)
    label = label.change_level(level=level)

    if table_name is None:
        label = _predict_simple(
            image=image,
            label=label,
            channel=channel,
            prediction_model=prediction_model,
            segmentation_model=segmentation_model,
        )
    else:
        label = _predict_with_roi(
            ngff_image=ngff_image,
            image=image,
            label=label,
            channel=channel,
            prediction_model=prediction_model,
            segmentation_model=segmentation_model,
            table_name=table_name,
        )
    label.consolidate()


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(task_function=plantseg_workflow)
