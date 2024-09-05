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

* The scrip might require some small modifications.

```bash
VERSION="v0.1.0" # Version of the package to install (by default the latest version)
COMMMAND="mamba" # Command to use to create the environment (mamba or conda) 
CUDA="12.1" # Available options: 12.1, 11.8 or CPU (default is 12.1)
# Location of the environment
# If ENVPREFIX is not NULL, the environment will be created with the prefix $ENVPREFIX/$ENVNAME 
# If ENVPREFIX is NULL, the environment will be created in the default location
ENVPREFIX="NULL" 
```

* Install the package using the installation script
  
```bash
bash create_env_script.sh
```

The installation script will create a conda environment with the name `fractal-plantseg-tasks` and install the package in the environment. It will also download the correct `__FRACTAL_MANIFEST__.json` file.

* In the fractal web interface add the task to the workflow as a "local env" task.
* Plantseg will download the necessary models on the first run. The default location for the models and data is `~/.plantseg_models`. If your system has a very limited home directory, you can set the environment variable `PLANTSEG_HOME` to a different location.
