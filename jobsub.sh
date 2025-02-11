#!/bin/bash
#SBATCH --job-name=p8q1u1
#SBATCH --error=p8q1u1.%J.err
#SBATCH --output=p8q1u1.%J.out
#SBATCH --partition=testp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=72:00:00

# Display help message if "--help" is passed
if [[ "$1" == "--help" ]]; then
    echo "Usage: $(basename "$0") [PREFIX] [EPOCHS]"
    echo ""
    echo "Arguments:"
    echo "  PREFIX    Optional. Prefix for the run. Default: p05"
    echo "  EPOCHS    Optional. Number of epochs for training. Default: 101"
    exit 0
fi

# Detect working directory and normalize paths
PWD=$(pwd)
if [[ "$PWD" == /nlsasfs/* ]]; then
    WRKDIR="$PWD"
elif [[ "$PWD" == /mnt/* ]]; then
    WRKDIR="/nlsasfs${PWD#/mnt}"
else
    echo "Error: Cannot determine correct path mapping for $PWD"
    exit 1
fi

####################### EDIT BELOW #######################

SCRIPTNAME='main.py'
# Assign arguments with default values
PREFIX="${1:-p08a}"   # Default: p08
EPOCHS="${2:-101}"   # Default: 101

####################### EDIT ABOVE #######################

echo "Initiating ${PREFIX} for ${EPOCHS} epochs ..."
echo "Working directory set to: $WRKDIR"
echo ""

# Check if main.py exists in the normalized path
if [[ ! -f "$WRKDIR/$SCRIPTNAME" ]]; then
    echo "Error: Cannot find $SCRIPTNAME in $WRKDIR"
    exit 1
fi

# Activate the conda environment
source /nlsasfs/home/precipitation/midhunm/Conda/bin/activate tf2
conda list

# Execute the Python script
python3 "$WRKDIR/$SCRIPTNAME" \
        --pwd "$WRKDIR" \
        --prefix "$PREFIX" \
        --epochs "$EPOCHS" \

# Print completion message
echo "Job exit at $(date '+%Y-%m-%d %H:%M:%S')"
