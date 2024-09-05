"""Contains the list of tasks available to fractal."""

from fractal_tasks_core.dev.task_models import NonParallelTask, ParallelTask

TASK_LIST = [
    NonParallelTask(
        name="H5 Converter Task",
        executable="convert_h5_to_ome_zarr.py",
        meta={"cpus_per_task": 1, "mem": 8000},
    ),
    NonParallelTask(
        name="Tiff Converter Task",
        executable="convert_tiff_to_ome_zarr.py",
        meta={"cpus_per_task": 1, "mem": 8000},
    ),
    ParallelTask(
        name="PlantSeg Workflow Task",
        executable="plantseg_workflow.py",
        meta={"cpus_per_task": 1, "mem": 32000, "needs_gpu": True},
    ),
]
