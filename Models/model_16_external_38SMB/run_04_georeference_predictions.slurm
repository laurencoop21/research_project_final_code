#!/bin/bash

#SBATCH --job-name=model16_38SMB_georef
#SBATCH --partition=tpg-gpu-small
#SBATCH --qos=short
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

cd "/home3/gdyx28/Research Project/Research Project/iraq_tells_gis/Models/model_16_external_38SMB" || exit 1

PYTHON="/home3/gdyx28/iraq_tells_env/bin/python"

echo "Georeferencing Model 16 predictions for 38SMB"
echo "Running on: $(hostname)"
echo "Python: $PYTHON"


"$PYTHON" -u 04_georeference_predictions.py
