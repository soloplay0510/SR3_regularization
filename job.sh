#!/bin/bash 
#SBATCH -J celeb_v0_train_celeb_1run 
#SBATCH -o celeb_v0_train_celeb_1run.o%j 
#SBATCH -t 24:00:00
#SBATCH -N 1 -n 1
#SBATCH --gpus=1
source /project/mwang/zxu29/anaconda3/bin/activate 
conda activate pytorch
ml CUDA
nvidia-smi   
python -c "import torch; print(torch.cuda.is_available())"

python sr.py -p train -c config/celebahq.json
