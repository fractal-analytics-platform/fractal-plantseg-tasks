from src.plantseg_tasks.convert_h5_to_ome_zarr import convert_h5_to_ome_zarr

from plantseg_tasks.task_utils.converter_input_models import (
    CustomAxisInputModel,
    OMEZarrBuilderParams,
)

input_path = "/home/lcerrone/data/ovule_sample.h5"

ome_zarr_parameters = OMEZarrBuilderParams(
    number_multiscale=5, scaling_factor_XY=2, create_all_ome_axis=True
)

convert_h5_to_ome_zarr(
    zarr_urls=[],
    zarr_dir="/home/lcerrone/data/ome-zarr/",
    input_path=input_path,
    image_key="raw",
    image_layout="ZYX",
    label_key="label",
    new_image_key=None,
    new_label_key=None,
    custom_axis=CustomAxisInputModel(),
    ome_zarr_parameters=ome_zarr_parameters,
)
