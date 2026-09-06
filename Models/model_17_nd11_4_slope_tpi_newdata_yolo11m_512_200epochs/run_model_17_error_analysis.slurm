#!/bin/bash

#SBATCH --job-name=model17_error
#SBATCH --partition=tpg-gpu-small
#SBATCH --qos=short
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

cd "/home3/gdyx28/Research Project/Research Project/iraq_tells_gis/Models/model_17_nd11_4_slope_tpi_newdata_yolo11m_512_200epochs" || exit 1

PYTHON="/home3/gdyx28/iraq_tells_env/bin/python"

echo "Starting Model 17 error analysis"
echo "Running on: $(hostname)"
echo "Python: $PYTHON"

"$PYTHON" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'); assert torch.cuda.is_available(), 'CUDA is not available'"

"$PYTHON" -u 07_error_analysis.py
