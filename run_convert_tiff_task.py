from src.plantseg_tasks.convert_tiff_to_ome_zarr import convert_tiff_to_ome_zarr

convert_tiff_to_ome_zarr(
    zarr_urls=[],
    zarr_dir="./ome-zarr/",
    image_path="./sample_image.tif",
    image_layout="YX",
)
