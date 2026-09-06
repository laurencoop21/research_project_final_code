#!/bin/bash

#SBATCH --job-name=model16_38SMB_review_pkg
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

echo "Creating advisor review package for 38SMB"
echo "Running on: $(hostname)"
echo "Python: $PYTHON"


"$PYTHON" -u 09_make_advisor_review_package.py
