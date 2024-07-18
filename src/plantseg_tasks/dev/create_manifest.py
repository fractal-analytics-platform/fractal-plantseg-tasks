"""Generate JSON schemas for task arguments afresh, and write them
to the package manifest.
"""  # noqa: D205

from fractal_tasks_core.dev.create_manifest import create_manifest

custom_pydantic_models = [
    (
        "plantseg_tasks",
        "task_utils/converter_input_models.py",
        "CustomAxisInputModel",
    ),
    (
        "plantseg_tasks",
        "task_utils/converter_input_models.py",
        "AxisScaleModel",
    ),
    (
        "plantseg_tasks",
        "task_utils/converter_input_models.py",
        "OMEZarrBuilderParams",
    ),
    (
        "plantseg_tasks",
        "task_utils/ps_workflow_input_models.py",
        "PlantSegPredictionsModel",
    ),
    (
        "plantseg_tasks",
        "task_utils/ps_workflow_input_models.py",
        "PlantSegSegmentationModel",
    ),
]
if __name__ == "__main__":
    PACKAGE = "plantseg_tasks"
    create_manifest(package=PACKAGE, custom_pydantic_models=custom_pydantic_models)
