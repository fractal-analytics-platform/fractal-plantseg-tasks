VERSION="v0.1.0"
COMMMAND="mamba"
CUDA="12.1" # Available options: 12.1, 11.8 or CPU
# If ENVPREFIX is not NULL, the environment will be created with the prefix $ENVPREFIX/$ENVNAME 
# If ENVPREFIX is NULL, the environment will be created in the default location
ENVPREFIX="NULL" 

# Do NOT change the following lines
ENVNAME=plant-seg-$VERSION
PYTHON="python=3.11"
PLANTSEG="plant-seg=1.8.1"

if [ "$CUDA" == "CPU" ]; then
    PYTORCH_PACKAGE="pytorch cpuonly"
else
    PYTORCH_PACKAGE="pytorch pytorch-cuda=$CUDA"
fi

if [ "$ENVPREFIX" == "NULL" ]; then
    LOCATION="--name $ENVNAME"
else
    LOCATION="--prefix $ENVPREFIX/$ENVNAME"
fi

echo "Creating environment $ENVNAME with $PYTORCH_PACKAGE"
$COMMMAND run $COMMMAND create $LOCATION \
                               -c pytorch \
                               -c nvidia  \
                               -c conda-forge $PYTHON $PYTORCH_PACKAGE pyqt $PLANTSEG bioimageio.core \
                               --no-channel-priority -y

echo "Installing plantseg-tasks version $VERSION"
$COMMMAND run --name $ENVNAME pip install git+https://github.com/fractal-analytics-platform/fractal-plantseg-tasks@$VERSION


echo "Downloading the __FRACTAL_MANIFEST__.json file file"
curl -O https://raw.githubusercontent.com/fractal-analytics-platform/fractal-plantseg-tasks/$VERSION/__FRACTAL_MANIFEST__.json