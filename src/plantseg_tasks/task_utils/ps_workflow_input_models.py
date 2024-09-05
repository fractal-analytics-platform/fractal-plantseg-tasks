"""PlantSeg workflow UI input models."""

from typing import Optional, Literal

from plantseg.models.zoo import model_zoo
from pydantic import BaseModel, Field


DEVICE = Literal["cpu", "cuda"]
MODEL_POOL = Literal["PlantSegZoo", "BioImageIO", "LocalModel"]

_all_plantseg_models = model_zoo.list_models()
DEFAULT_MODEL = _all_plantseg_models[0]
PLANTSEG_MODEL = Literal[*_all_plantseg_models]  # type: ignore

_all_bioio_models = model_zoo.get_bioimageio_zoo_plantseg_model_names()
DEFAULT_BIOIO_MODEL = _all_bioio_models[0]
BIOIO_MODEL = Literal[*_all_bioio_models]  # type: ignore


class PlantSegPredictionsModel(BaseModel):
    """Input model for PlantSeg predictions.

    Args:
        model_source (MODEL_POOL): Define which of the following fields to use.
        plantsegzoo_name (PLANTSEG_MODEL): The model name from the PlantSeg Zoo.
            This field is only used if model_source is PlantSegZoo.
        bioimageio_name (BIOIO_MODEL): The model name from the BioImageIO Zoo.
            This field is only used if model_source is BioImageIO.
        local_model_path (str): The path to the local model.
            This field is only used if model_source is LocalModel.
        device (DEVICE): The device to use. Must be one of 'cpu', 'cuda'.
        patch (tuple[int, int, int]): The patch size.
        skip (bool): Whether to skip the predictions.
    """

    model_source: MODEL_POOL = "PlantSegZoo"
    plantsegzoo_name: PLANTSEG_MODEL = DEFAULT_MODEL  # type: ignore
    bioimageio_name: BIOIO_MODEL = DEFAULT_BIOIO_MODEL  # type: ignore
    local_model_path: Optional[str] = None
    device: Literal["cpu", "cuda"] = "cuda"
    patch: tuple[int, ...] = (80, 160, 160)
    skip: bool = False


SEGMENTATION_TYPE = Literal["gasp", "mutex_ws", "multicut", "dt_watershed"]


class PlantSegSegmentationModel(BaseModel):
    """Input model for PlantSeg segmentations.

    Args:
        ws_threshold (float): The threshold for the watershed.
        segmentation_type (SEGMENTATION_TYPE): The segmentation method to use.
            Must be one of 'gasp', 'mutex_ws', 'multicut', 'dt_watershed'.
        beta (float): The beta value.
        post_minsize (int): The minimum size.
        skip (bool): Whether to skip the segmentation.
    """

    ws_threshold: float = 0.5
    segmentation_type: SEGMENTATION_TYPE = Field(
        default="gasp", title="Segmentation Method"
    )
    beta: float = 0.6
    post_minsize: int = 100
