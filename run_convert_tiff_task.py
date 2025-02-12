from src.plantseg_tasks.convert_tiff_to_ome_zarr import convert_tiff_to_ome_zarr

convert_tiff_to_ome_zarr(
    zarr_urls=[],
    zarr_dir="./ome-zarr/",
    image_path="./2d_cells_apoptotic_nuclei_export.tiff",
    image_layout="YX",
)
