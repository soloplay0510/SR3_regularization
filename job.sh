#!/bin/bash 
#SBATCH -J celeb_v0_64_pretrain
#SBATCH -o ./logs/celeb_v0_64_pretrain.o%j 
#SBATCH -t 40:00:00
#SBATCH -N 1 -n 1
#SBATCH --gpus=1
source /project/mwang/zxu29/anaconda3/bin/activate 
conda activate pytorch_user         
#ml CUDA
#nvidia-smi
#python -c "import torch; print(torch.cuda.is_available())"
python sr.py -p pretrain -c config/celebahq.json
# python sr.py -p train -c config/celebahq.json
#python infer.py -c config/celebahq_test.json --n_runs 5 --seed 1234


#python batch_eval_checkpoints.py \
#  --config config/celebahq_vgg.json \
#  --checkpoint-dir experiments/tv0_4x_vgg_later_260331_100658/checkpoint \
#  --start-index 1 \
#  --index-step 10 \
#  --pick first \
#  --limit 8 \
#  --n-run 100 \
#  --gpu-ids 0 \
#  --mkl-threading-layer GNU \
#  --output-dir plots/checkpoint_eval_vgg1E-1_later

#python /project/mwang/mwang55/SR3_regularization/plot_sample_across_checkpoints.py \
#  --summary-csvs \
#  /project/mwang/mwang55/SR3_regularization/plots/checkpoint_eval_every10_tv0_x4_VGG1E-1/checkpoint_eval_summary.csv \
#  /project/mwang/mwang55/SR3_regularization/plots/checkpoint_eval_every10_tv0_x4/checkpoint_eval_summary.csv \
#  /project/mwang/mwang55/SR3_regularization/plots/checkpoint_eval_every10_tv0_x4/checkpoint_eval_summary.csv \
#  --series-labels "vgg1E-1,no vgg, vgg1E-1_CNN" \
#  --metric psnr \
#  --sample-indices 1,2,3 \
#  --output-dir /project/mwang/mwang55/SR3_regularization/plots/checkpoint_eval_compare_vgg_vs_novgg_4x
#123456
# Resize to get 16×16 LR_IMGS and 128×128 HR_IMGS, then prepare 128×128 Fake SR_IMGS by bicubic interpolation
#python data/prepare_data.py  --path  /project/mwang/mwang55/SR3_regularization/dataset/celebahq_16_128/hr_128 --out /project/mwang/mwang55/SR3_regularization/dataset/celebahq --size 32,128 -l