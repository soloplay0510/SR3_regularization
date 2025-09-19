#!/bin/bash 
#SBATCH -J celeb_v0_stdrelu_train_image
#SBATCH -o ./logs/celeb_v0_stdrelu_train_image.o%j 
#SBATCH -t 40:00:00
#SBATCH -N 1 -n 1
#SBATCH --gpus=1
source /project/mwang/zxu29/anaconda3/bin/activate 
conda activate pytorch_user         
ml CUDA
nvidia-smi   
python -c "import torch; print(torch.cuda.is_available())"

python sr.py -p train -c config/celebahq.json
