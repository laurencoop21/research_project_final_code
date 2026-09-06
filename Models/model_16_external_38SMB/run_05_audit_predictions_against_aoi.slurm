#!/bin/bash

#SBATCH --job-name=model16_38SMB_aoi_audit
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

echo "Auditing Model 16 predictions against the 38SMB external AOI"
echo "Running on: $(hostname)"
echo "Python: $PYTHON"


"$PYTHON" -u 05_audit_predictions_against_aoi.py
