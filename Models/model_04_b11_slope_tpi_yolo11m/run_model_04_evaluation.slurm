#!/bin/bash

#SBATCH --job-name=model04_test
#SBATCH --partition=tpg-gpu-small
#SBATCH --qos=short
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=evaluation-%j.out
#SBATCH --error=evaluation-%j.err

cd "/home3/gdyx28/Research Project/Research Project/iraq_tells_gis/Models/model_04_b11_slope_tpi_yolo11m" || exit 1

PYTHON="/home3/gdyx28/iraq_tells_env/bin/python"

echo "Starting Model 4 test evaluation"
echo "Running on: $(hostname)"

"$PYTHON" -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'); assert torch.cuda.is_available()"

"$PYTHON" -u 06_evaluate_test.py
