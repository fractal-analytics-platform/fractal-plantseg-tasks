"""Main PlantSeg processing functions."""

from functools import partial
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import zarr
from fractal_tasks_core.ngff.specs import NgffImageMeta
from fractal_tasks_core.ngff.zarr_utils import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.utils import logger
from plantseg.predictions.functional import unet_predictions
from plantseg.segmentation.functional import dt_watershed, gasp, multicut, mutex_ws

from plantseg_tasks.task_utils.ps_workflow_input_models import (
    PlantSegPredictionsModel,
    PlantSegSegmentationModel,
)


def _save_multiscale_image(
    zarr_url: Path,
    image: np.ndarray,
    metadata: NgffImageMeta,
    aggregation_function: Optional[Callable] = None,
    mode="a",
):
    ome_zarr = zarr.open(str(zarr_url), mode=mode)
    assert isinstance(ome_zarr, zarr.Group), "Zarr image must be a group."
    ome_zarr.create_dataset("0", data=image, dimension_separator="/")
    ome_zarr.attrs.update(metadata.model_dump(exclude_none=True))
    chunksize = ome_zarr.get("0")
    if chunksize is not None:
        chunksize = chunksize.chunks
    else:
        chunksize = None

    if aggregation_function is None:
        aggregation_function = np.mean

    build_pyramid(
        zarrurl=zarr_url,
        overwrite=True,
        chunksize=chunksize,
        num_levels=metadata.num_levels,
        coarsening_xy=metadata.coarsening_xy,
        aggregation_function=aggregation_function,
    )


def _load_zarr(
    zarr_url: str, channel, level: int = 0
) -> tuple[np.ndarray, NgffImageMeta]:
    raw_image = zarr.open(zarr_url + f"/{level}")
    metadata = load_NgffImageMeta(zarr_url)

    # This a workaround before the new OME-Zarr io implementation
    axis = metadata.multiscales[0].axes
    slices = []
    for _, ax in enumerate(axis):
        if ax.type == "c" or ax.type == "channel":
            _slice = channel
        else:
            _slice = slice(None)
        slices.append(_slice)

    assert isinstance(raw_image, zarr.Array), "Raw image must be a zarr array."
    raw_image = raw_image[tuple(slices)]
    return raw_image, metadata


def plantseg_predictions(
    raw_image: np.ndarray,
    prediction_model: PlantSegPredictionsModel,
):
    """PlantSeg predictions function.

    Args:
        raw_image: The raw image to predict.
        prediction_model: The prediction model.
    """
    if prediction_model.model_source == "PlantSegZoo":
        model_name = prediction_model.plantsegzoo_name
        model_id = None
        config_path = None
        model_weights_path = None
    elif prediction_model.model_source == "BioImageIO":
        model_name = None
        model_id = prediction_model.bioimageio_name
        config_path = None
        model_weights_path = None
    elif prediction_model.model_source == "LocalModel":
        model_name = None
        model_id = None
        config_path = f"{prediction_model.local_model_path}/config.yaml"
        model_weights_path = f"{prediction_model.local_model_path}/model.pth"

    predictions = unet_predictions(
        raw_image,
        model_name=model_name,
        model_id=model_id,
        patch=prediction_model.patch,
        single_batch_mode=True,
        device=prediction_model.device,
        model_update=False,
        disable_tqdm=True,
        handle_multichannel=False,
        config_path=config_path,
        model_weights_path=model_weights_path,
    )
    return predictions


def plantseg_segmentation(
    prediction: np.ndarray,
    segmentation_model: PlantSegSegmentationModel,
) -> str:
    """PlantSeg segmentation function.

    Args:
        prediction: The prediction to segment.
        segmentation_model: The segmentation model.
    """
    # run distance transform watershed
    segmentation = dt_watershed(prediction, threshold=segmentation_model.ws_threshold)

    if segmentation_model.segmentation_type == "gasp":
        segmentation_func = partial(gasp, n_threads=1)

    elif segmentation_model.segmentation_type == "mutex_ws":
        segmentation_func = partial(mutex_ws, n_threads=1)

    elif segmentation_model.segmentation_type == "multicut":
        segmentation_func = multicut

    elif segmentation_model.segmentation_type == "dt_watershed":
        # avoid re-running dt_watershed
        def segmentation_func(*args, **kwargs):
            return segmentation
    else:
        raise ValueError("Invalid segmentation type.")

    segmentation = segmentation_func(
        boundary_pmaps=prediction,
        superpixels=segmentation,
        beta=segmentation_model.beta,
        post_minsize=segmentation_model.post_minsize,
    )
    return segmentation


def plantseg_standard_workflow(
    image: np.ndarray,
    prediction_model: PlantSegPredictionsModel,
    segmentation_model: PlantSegPredictionsModel,
) -> np.ndarray:
    """Full PlantSeg workflow.

    This function runs the full PlantSeg workflow on a OME-Zarr file.
    The workflow consists of two main steps:
    1. Predictions
    2. Segmentation

    Args:
        image: The image to process.
        prediction_model: The prediction model.
        segmentation_model: The segmentation model.
    """
    if prediction_model.skip:
        predictions = image
        logger.info("Skipping predictions step.")
    else:
        predictions = plantseg_predictions(image, prediction_model)
        logger.info("Predictions step completed.")
    segmentation = plantseg_segmentation(predictions, segmentation_model)
    logger.info("Segmentation step completed.")
    return segmentation
