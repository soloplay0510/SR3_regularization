####combine multiple run log for output
cat \
    ./experiments/tv0_260108_230910/logs/val.log \
    > ./plots/tv0.log
cat \
    ./experiments/tv0_260108_230910/logs/train.log \
    > ./plots/tv0_train.log

cat \
    ./experiments/tv0_vgg_1E-1_260108_225707/logs/train.log \
    > ./plots/vgg_train.log
cat \
    ./experiments/tv0_vgg_1E-1_260108_225707/logs/val.log \
    > ./plots/vgg_val.log

cat \
    ./experiments/tv0_vgg_1E0_260126_164937/logs/train.log \
    > ./plots/vgg_1E0_train.log
cat \
    ./experiments/tv0_vgg_1E0_260126_164937/logs/val.log \
    > ./plots/vgg_1E0_val.log

cat \
    ./experiments/tv0_vgg_1E1_260126_164838/logs/train.log \
    > ./plots/vgg_1E1_train.log
cat \
    ./experiments/tv0_vgg_1E1_260126_164838/logs/val.log \
    > ./plots/vgg_1E1_val.log


cat \
    ./experiments/tv0_vgg_1E3_260112_185050/logs/train.log \
    > ./plots/vgg_1E3_train.log
cat \
    ./experiments/tv0_vgg_1E3_260112_185050/logs/val.log \
    > ./plots/vgg_1E3_val.log

cat \
    ./experiments/tv0_vgg_1E5_260112_184912/logs/train.log \
    > ./plots/vgg_1E5_train.log
cat \
    ./experiments/tv0_vgg_1E5_260112_184912/logs/val.log \
    > ./plots/vgg_1E5_val.log

cat \
    ./experiments/tv0_vgg_1E6_260112_184808/logs/train.log \
    > ./plots/vgg_1E6_train.log
cat \
    ./experiments/tv0_vgg_1E6_260112_184808/logs/val.log \
    > ./plots/vgg_1E6_val.log

cat \
    ./experiments/tv0_vgg_1E7_260112_184850/logs/train.log \
    > ./plots/vgg_1E7_train.log
cat \
    ./experiments/tv0_vgg_1E7_260112_184850/logs/val.log \
    > ./plots/vgg_1E7_val.log



cat \
    ./experiments/tv0_vgg_1E-1_regularizer_260217_011004/logs/train.log \
    > ./plots/vgg_1E-1_train_reg.log
cat \
    ./experiments/tv0_vgg_1E-1_regularizer_260217_011004/logs/val.log \
    > ./plots/vgg_1E-1_val_reg.log    
#### Curves of metrics
# Compare three runs; full grid (epoch & iteration), mark best, smooth over 5 points

python plot_evaluation.py ./plots/vgg_val.log ./plots/vgg_1E0_val.log ./plots/vgg_1E1_val.log ./plots/vgg_1E3_val.log ./plots/vgg_1E5_val.log ./plots/vgg_1E6_val.log ./plots/vgg_1E7_val.log ./plots/tv0.log --labels vgg_1E-1 vgg_1E0 vgg_1E1 vgg_1E3 vgg_1E5 vgg_1E6 vgg_1E7 swish --smooth 5 --output plots/vgg_swish_comp.png --epoch-only


python plot_evaluation.py ./plots/vgg_val.log ./plots/vgg_1E-1_val_reg.log ./plots/tv0.log --labels vgg_1E-1 vgg_1E-1_reg no_vgg --smooth 5 --output plots/vgg_reg_comp.png --epoch-only

# Dual-y single chart (epoch vs PSNR/SSIM together)
# python plot_evaluation.py ./plots/tv0_relu.log ./plots/tv0.log --dual-y --output plots/relu_swish.png --metrics psnr --smooth 5 --label relu swish --max-epoch 140


# Also dump parsed points to CSV
python plot_evaluation.py ./plots/tv0.log --out-csv plots/swish.csv
python plot_evaluation.py ./plots/vgg_val.log --out-csv plots/vgg_swish.csv

#### Images of evaluations
# Default (recommended): compare the "latest"/max leadnum across folders, per idx


python training_eval_images.py ./experiments/tv0_vgg_1E-1_260108_225707/results  --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix vgg01_swish_ --separate-groups

python training_eval_images.py ./experiments/tv0_260108_230910/results  --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix swish_ --separate-groups

python training_eval_images.py ./experiments/tv0_vgg_1E-1_regularizer_260217_011004/results  --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix vgg01_reg_swish_ --separate-groups


##### Plot of losses with smoothing
# single log
# metrics:  total loss_noise loss_TV1 loss_TV2 loss_TVF loss_wave_l1

python plot_losses.py  plots/vgg_train.log \
  --metrics total loss_noise loss_vgg \
  --labels vgg_swish \
  --smooth 1000 --out_png plots/vgg_loss_curves.png \
  --xaxis epoch \
#   --yscale log

python plot_losses.py  plots/vgg_1E0_train.log \
  --metrics total loss_noise loss_vgg \
  --labels vgg_swish \
  --smooth 1000 --out_png plots/vgg1E0_loss_curves.png \
  --xaxis epoch \


python plot_losses.py  plots/vgg_1E-1_train_reg.log \
  --metrics total loss_noise loss_vgg loss_trainable_regularizer  \
  --labels vgg_reg_swish \
  --smooth 1000 --out_png plots/vgg1E-1_reg_loss_curves.png \
  --xaxis epoch \
# compare two logs


# python plot_losses.py plots/vgg_train.log plots/vgg_1E0_train.log plots/vgg_1E1_train.log plots/vgg_1E3_train.log plots/vgg_1E5_train.log plots/vgg_1E6_train.log plots/vgg_1E7_train.log plots/tv0_train.log \
#   --labels vgg_1E-1_swish vgg_1E0_swish vgg_1E1_swish vgg_1E3_swish vgg_1E5_swish vgg_1E6_swish vgg_1E7_swish swish \
#   --metrics loss_noise \
#   --xaxis epoch --smooth 1000 --out_png plots/vgg_64_swish_loss_compare.png \
#   --yscale log \


# python plot_losses.py plots/vgg_train.log plots/vgg_1E0_train.log plots/vgg_1E1_train.log plots/vgg_1E3_train.log plots/vgg_1E5_train.log plots/vgg_1E6_train.log plots/vgg_1E7_train.log plots/tv0_train.log \
#   --labels vgg_1E-1_swish vgg_1E0_swish vgg_1E1_swish vgg_1E3_swish vgg_1E5_swish vgg_1E6_swish vgg_1E7_swish swish \
#   --metrics loss_vgg \
#   --xaxis epoch --smooth 1000 --out_png plots/vgg_64_swish_loss_vgg_compare.png \
#   --yscale log \

python plot_losses.py plots/vgg_train.log plots/vgg_1E-1_train_reg.log plots/tv0_train.log \
  --labels vgg_1E-1 vgg_1E-1_reg swish \
  --metrics loss_noise \
  --xaxis epoch --smooth 1000 --out_png plots/vgg_64_swish_loss_vgg_reg_compare.png \
  --yscale log \

python plot_losses.py plots/vgg_train.log plots/vgg_1E-1_train_reg.log plots/tv0_train.log \
  --labels vgg_1E-1 vgg_1E-1_reg swish \
  --metrics loss_vgg \
  --xaxis epoch --smooth 1000 --out_png plots/vgg_64_swish_loss_vgg_reg_compare_2.png \
  --yscale log \