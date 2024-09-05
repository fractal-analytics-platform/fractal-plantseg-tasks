VERSION="0.1.0"
COMMMAND="mamba"
CUDA="CPU" # Available options: 12.1, 11.8 or CPU

# Do NOT change the following lines

ENVNAME=plant-seg-$VERSION

if [ "$CUDA" == "CPU" ]; then
    PYTORCH_PACKAGE="pytorch cpuonly"
else
    PYTORCH_PACKAGE=pytorch pytorch-cuda=$CUDA
fi

echo "Creating environment $ENVNAME with $PYTORCH_PACKAGE"
$COMMMAND run $COMMMAND create --name $ENVNAME \
                               -c pytorch \
                               -c nvidia  \
                               -c conda-forge python=3.11 $PYTORCH_PACKAGE pyqt plant-seg==1.8.1 bioimageio.core \
                               --no-channel-priority -y

echo "Installing plantseg-tasks version $VERSION"
$COMMMAND run --name $ENVNAME pip install git+https://github.com/fractal-analytics-platform/fractal-plantseg-tasks@$VERSION