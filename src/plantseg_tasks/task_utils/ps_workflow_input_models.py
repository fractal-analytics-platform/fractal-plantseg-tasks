"""PlantSeg workflow UI input models."""

from enum import StrEnum

from plantseg.models.zoo import model_zoo
from pydantic.v1 import BaseModel, Field


class Device(StrEnum):
    """Device to use for CNN predictions."""

    cpu = "cpu"
    cuda = "cuda"


DynamicallyGeneratedModel = StrEnum(
    "DynamicallyGeneratedModel",
    {model_name: model_name for model_name in model_zoo.list_models()},
)


DynamicBioIOModels = StrEnum(
    "DynamicBioIOModels",
    {
        model_name: model_name
        for model_name in model_zoo.get_bioimageio_zoo_all_model_names()
    },
)

DynamicallyGeneratedModel.__doc__ = "Select a model from the PlantSeg Zoo."
DynamicBioIOModels.__doc__ = "Select a model from the BioImageIO Zoo."


class ModelsPool(StrEnum):
    """Select if the model is sourced from PlantSegZoo or BioImageIO."""

    PlantSegZoo = "PlantSegZoo"
    BioImageIO = "BioImageIO"


class PlantSegPredictionsModel(BaseModel):
    """Input model for PlantSeg predictions.

    Args:
        model_source (ModelsPool): The source of the model.
        plantsegzoo_name (DynamicallyGeneratedModel): The model name from
            the PlantSeg Zoo.
        bioimageio_name (DynamicBioIOModels): The model name from the BioImageIO Zoo.
        device (Device): The device to use for predictions.
        patch (tuple[int, int, int]): The patch size.
        save_results (bool): Whether to save the results.
        skip (bool): Whether to skip the predictions.
    """

    model_source: ModelsPool = ModelsPool.PlantSegZoo
    plantsegzoo_name: DynamicallyGeneratedModel = model_zoo.list_models()[0]  # type: ignore
    bioimageio_name: DynamicBioIOModels = (  # type: ignore
        model_zoo.get_bioimageio_zoo_all_model_names()[0]
    )
    device: Device = Device.cuda
    patch: tuple[int, int, int] = (80, 160, 160)
    save_results: bool = False
    skip: bool = False


class SegmentationType(StrEnum):
    """PlantSeg segmentation function to use.

    Implemented segmentation types:
        - gasp: A generalised agglomerative algorithm,
            used by default fairly fast and robust.
        - mutex_ws: Mutex watershed. Very fast, but only
            works well on very clean predictions.
        - multicut: Multicut segmentation.
            Slower than gasp, but more robust.
        - dt_watershed: Distance transform watershed.
            Very fast, but tends to oversegment a lot.
    """

    gasp = "gasp"
    mutex_ws = "mutex_ws"
    multicut = "multicut"
    dt_watershed = "dt_watershed"


class PlantSegSegmentationModel(BaseModel):
    """Input model for PlantSeg segmentations.

    Args:
        ws_threshold (float): The threshold for the watershed.
        segmentation_type (SegmentationType): The segmentation method to use.
        beta (float): The beta value.
        post_minsize (int): The minimum size.
        skip (bool): Whether to skip the segmentation.
    """

    ws_threshold: float = 0.5
    segmentation_type: SegmentationType = Field(
        default=SegmentationType.gasp, title="Segmentation Method"
    )
    beta: float = 0.6
    post_minsize: int = 100
    skip: bool = False
