VERSION="0.1.0"
COMMMAND="mamba"
CUDA = "12.1"

echo $(which python)

ENVNAME=plant-seg-$VERSION
$COMMMAND run $COMMMAND create --name $ENVNAME \
                               -c pytorch \
                               -c nvidia  \
                               -c conda-forge \ 
                               pytorch pytorch-cuda=$CUDA pyqt plant-seg bioimageio.core \
                               --no-channel-priority \
                               -y

$COMMMAND run --name $ENVNAME pip install git+https://github.com/fractal-analytics-platform/fractal-plantseg-tasks@$VERSION