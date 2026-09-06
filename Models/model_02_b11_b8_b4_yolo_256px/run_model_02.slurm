#!/bin/bash

#SBATCH --job-name=model02_yolo_256
#SBATCH --partition=tpg-gpu-small
#SBATCH --qos=short
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=model02-%j.out
#SBATCH --error=model02-%j.err

cd "/home3/gdyx28/Research Project/Research Project/iraq_tells_gis/Models/model_02_b11_b8_b4_yolo_256px" || exit 1

PYTHON="/home3/gdyx28/iraq_tells_env/bin/python"

echo "Starting Model 2: 256-pixel YOLO"
echo "Running on: $(hostname)"
echo "Python: $PYTHON"

"$PYTHON" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'); assert torch.cuda.is_available(), 'CUDA is not available'"

"$PYTHON" 05_train_model.py