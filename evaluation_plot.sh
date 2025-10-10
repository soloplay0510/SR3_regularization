####combine multiple run log for output
cat \
    ./experiments/celeb_tv0_250827_231027/logs/val.log \
    > ./plots/tv0.log
cat \
    ./experiments/celeb_tv0_250827_231027/logs/train.log \
    > ./plots/tv0_train.log

cat \
    ./experiments/celeb_tv0_tvf_250830_110842/logs/train.log \
    > ./plots/tvf_train.log
cat \
    ./experiments/celeb_tv0_tvf_250830_110842/logs/val.log \
    > ./plots/tvf_val.log

cat \
    ./experiments/celeb_tv0_wavelet_250830_105558/logs/train.log \
    > ./plots/wave_train.log
cat \
    ./experiments/celeb_tv0_wavelet_250830_105558/logs/val.log \
    > ./plots/wave_val.log
cat \
    ./experiments/tv0_CNN_250922_123148/logs/train.log \
    > ./plots/cnn_train.log
cat \
   ./experiments/tv0_CNN_250922_123148/logs/val.log \
    > ./plots/cnn_val.log

cat \
    ./experiments/tv0_64_CNN_251006_101148/logs/train.log \
    > ./plots/cnn_64_train.log
cat \
   ./experiments/tv0_64_CNN_251006_101148/logs/val.log \
    > ./plots/cnn_64_val.log
cat \
    ./experiments/tv0_64_251008_112202/logs/train.log \
    > ./plots/tv0_64_train.log
cat \
   ./experiments/tv0_64_251008_112202/logs/val.log \
    > ./plots/64_val.log



#### Curves of metrics
# Compare three runs; full grid (epoch & iteration), mark best, smooth over 5 points
python plot_evaluation.py ./plots/wave_val.log ./plots/tvf_val.log ./plots/tv0.log --labels wave_swish tvf_swish swish --smooth 5 --output plots/wave_tvf_swish_comp.png --epoch-only
python plot_evaluation.py ./plots/cnn_val.log ./plots/tv0.log --labels cnn_swish swish --smooth 5 --output plots/cnn_swish_comp.png --epoch-only
python plot_evaluation.py ./plots/cnn_64_val.log ./plots/64_val.log --labels cnn_64_swish 64_swish --smooth 5 --output plots/cnn_64_swish_comp.png --epoch-only

# Dual-y single chart (epoch vs PSNR/SSIM together)
# python plot_evaluation.py ./plots/tv0_relu.log ./plots/tv0.log --dual-y --output plots/relu_swish.png --metrics psnr --smooth 5 --label relu swish --max-epoch 140

python plot_evaluation.py  ./plots/wave_val.log ./plots/tvf_val.log ./plots/tv0.log --dual-y --output plots/wave_tvf_swish.png --metrics psnr --smooth 5 --label wave_swish tvf_swish swish --max-epoch 140
# Also dump parsed points to CSV
python plot_evaluation.py ./plots/tv0.log --out-csv plots/swish.csv
python plot_evaluation.py ./plots/tvf_val.log --out-csv plots/tvf_swish.csv
python plot_evaluation.py ./plots/wave_val.log --out-csv plots/wave_swish.csv

#### Images of evaluations
# Default (recommended): compare the "latest"/max leadnum across folders, per idx

python training_eval_images.py ./experiments/celeb_tv0_250827_231027/results --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix swish_ --separate-groups
python training_eval_images.py ./experiments/celeb_tv0_tvf_250829_165656/results  --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix tvf_swish_ --separate-groups

python training_eval_images.py ./experiments/celeb_tv0_wavelet_250830_105558/results  --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix wave_swish_ --separate-groups

python training_eval_images.py ./experiments/tv0_CNN_250922_123148/results  --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix CNN_swish_ --separate-groups

python training_eval_images.py ./experiments/tv0_64_CNN_251006_101148/results  --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix CNN_64_swish_ --separate-groups

python training_eval_images.py ./experiments/tv0_64_251008_112202/results  --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix 64_swish_ --separate-groups
##### Plot of losses with smoothing
# single log
# metrics:  total loss_noise loss_TV1 loss_TV2 loss_TVF loss_wave_l1

python plot_losses.py  plots/tvf_train.log \
  --metrics total loss_noise loss_TVF \
  --labels tvf_swish \
  --smooth 1000 --out_png plots/tvf_loss_curves.png \
  --xaxis epoch \
#   --yscale log
python plot_losses.py  plots/wave_train.log \
  --metrics total loss_noise loss_wave_l1 \
  --labels wave_swish \
  --smooth 1000 --out_png plots/wave_loss_curves.png \
  --xaxis epoch \

python plot_losses.py  plots/cnn_64_train.log \
  --metrics total loss_noise loss_TV1 \
  --labels cnn_64 \
  --smooth 1000 --out_png plots/cnn_64_loss_curves.png \
  --xaxis epoch \

python plot_losses.py  plots/tv0_64_train.log \
  --metrics total loss_noise loss_TV1 \
  --labels tv0_64 \
  --smooth 1000 --out_png plots/tv0_64_loss_curves.png \
  --xaxis epoch \

# compare two logs

python plot_losses.py plots/wave_train.log plots/tvf_train.log  plots/tv0_train.log \
  --labels wave_swish tvf_swish swish \
  --metrics loss_noise \
  --xaxis epoch --smooth 1000 --out_png plots/wave_tvf_swish_loss_compare.png \
  --yscale log \

python plot_losses.py plots/cnn_train.log  plots/tv0_train.log \
  --labels cnn_swish swish \
  --metrics loss_noise \
  --xaxis epoch --smooth 1000 --out_png plots/cnn_swish_loss_compare.png \
  --yscale log \

python plot_losses.py plots/cnn_64_train.log  plots/tv0_64_train.log \
  --labels cnn_swish_64 swish_64 \
  --metrics loss_noise \
  --xaxis epoch --smooth 1000 --out_png plots/cnn_64_swish_loss_compare.png \
  --yscale log \