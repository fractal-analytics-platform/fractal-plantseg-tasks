# fractal-plantseg-tasks

Collection of Fractal task with the PlantSeg segmentation pipeline.

## Tasks

1. **Plantseg Workflow**: The main PlantSeg segmentation pipeline. For detailed information on PlantSeg, please refer to the [PlantSeg repository](https://github.com/kreshuklab/plant-seg).
2. **Tiff Converters**: A Basic converter to convert a simple tiff file in OME-Zarr format.
3. **HDF5 Converters**: A Basic converter to convert a simple hdf5 file in OME-Zarr format.

## Installation and Deployment

* Install the `mamba` package manager

* Download the installation script from this repository

```bash
curl -O https://raw.githubusercontent.com/fractal-analytics-platform/fractal-plantseg-tasks/main/create_env_script.sh
```

* Edit the installation script to set a custom location for the conda environment and for selecting a specific version of the Fractal-PlantSeg-Tasks package

* Install the package using the installation script
  
```bash
bash create_env_script.sh
```

The installation script will create a conda environment with the name `fractal-plantseg-tasks` and install the package in the environment. It will also download the `__FRACTAL_MANIFEST__.json`.

* In the fractal web interface add the task to the workflow as a "local" task.
