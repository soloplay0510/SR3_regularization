####combine multiple run log for output
 cat \
    ./experiments/celeb_tv0_stdrelu_250824_162052/logs/val.log \
    ./experiments/celeb_tv0_stdrelu_250825_151239/logs/val.log \
    > ./plots/tv0_relu.log

 cat \
    ./experiments/celeb_tv0_stdrelu_250824_162052/logs/train.log \
    ./experiments/celeb_tv0_stdrelu_250825_151239/logs/train.log \
    > ./plots/tv0_relu_train.log

 cat \
    ./experiments/celeb_tv0_leakystdrelu_250825_150651/logs/val.log \
    ./experiments/celeb_tv0_stdleakyrelu_250827_230725/logs/val.log \
    > ./plots/tv0_leaky_relu.log
 cat \
    ./experiments/celeb_tv0_leakystdrelu_250825_150651/logs/train.log \
    ./experiments/celeb_tv0_stdleakyrelu_250827_230725/logs/train.log \
    > ./plots/tv0_leaky_relu_train.log
 cat \
    ./experiments/celeb_tv0_250827_231027/logs/val.log \
    > ./plots/tv0.log
cat \
    ./experiments/celeb_tv0_250827_231027/logs/train.log \
    > ./plots/tv0_train.log

cat \
    ./experiments/celeb_tv0_tvf_250829_165656/logs/train.log \
    > ./plots/tvf_train.log
cat \
    ./experiments/celeb_tv0_tvf_250829_165656/logs/val.log \
    > ./plots/tvf_val.log
#### Curves of metrics
# Compare three runs; full grid (epoch & iteration), mark best, smooth over 5 points
python plot_evaluation.py ./plots/tv0_relu.log ./plots/tv0_leaky_relu.log ./plots/tv0.log --labels relu leaky_relu swish --smooth 5 --output plots/relu_leaky_relu_swish_comp.png --epoch-only
python plot_evaluation.py ./plots/tvf_val.log ./plots/tv0.log --labels tvf_swish swish --smooth 5 --output plots/tvf_swish_comp.png --epoch-only
# Dual-y single chart (epoch vs PSNR/SSIM together)
python plot_evaluation.py ./plots/tv0_relu.log ./plots/tv0.log --dual-y --output plots/relu_swish.png --metrics psnr --smooth 5 --label relu swish --max-epoch 140

python plot_evaluation.py ./plots/tvf_val.log ./plots/tv0.log --dual-y --output plots/tvf_swish.png --metrics psnr --smooth 5 --label tvf_swish swish --max-epoch 140
# Also dump parsed points to CSV
python plot_evaluation.py ./plots/tv0_relu.log --out-csv plots/relu.csv
python plot_evaluation.py ./plots/tv0_leaky_relu.log --out-csv plots/leaky_relu.csv
python plot_evaluation.py ./plots/tv0.log --out-csv plots/swish.csv
python plot_evaluation.py ./plots/tvf_val.log --out-csv plots/tvf_swish.csv

#### Images of evaluations
# Default (recommended): compare the "latest"/max leadnum across folders, per idx
python training_eval_images.py ./experiments/celeb_tv0_leakystdrelu_250825_150651/results ./experiments/celeb_tv0_stdleakyrelu_250827_230725/results  --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix leaky_relu_ --separate-groups
python training_eval_images.py ./experiments/celeb_tv0_stdrelu_250824_162052/results ./experiments/celeb_tv0_stdrelu_250825_151239/results --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix relu_ --separate-groups
python training_eval_images.py ./experiments/celeb_tv0_250827_231027/results --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix swish_ --separate-groups
python training_eval_images.py ./experiments/celeb_tv0_tvf_250829_165656/results  --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix tvf_swish_ --separate-groups



##### Plot of losses with smoothing
# single log
# metrics:  total loss_noise loss_TV1 loss_TV2 loss_TVF loss_wave_l1
python plot_losses.py  plots/tv0_relu_train.log \
  --metrics total loss_noise loss_TV1 loss_TV2 loss_TVF loss_wave_l1 \
  --labels relu \
  --smooth 1000 --out_png plots/relu_loss_curves.png \
  --xaxis epoch \
#   --yscale log

python plot_losses.py  plots/tvf_train.log \
  --metrics total loss_noise loss_TVF \
  --labels relu \
  --smooth 1000 --out_png plots/tvf_loss_curves.png \
  --xaxis epoch \
#   --yscale log
# compare two logs
python plot_losses.py plots/tv0_relu_train.log plots/tv0_leaky_relu_train.log plots/tv0_train.log\
  --labels relu leaky_relu swish \
  --metrics total \
  --xaxis iter --smooth 1000 --out_png plots/loss_compare_swish_leaky_relu.png \
  --yscale log \
#   --iter-min 20000 --iter-max 550000


python plot_losses.py plots/tvf_train.log  plots/tv0_train.log \
  --labels tvf_swish swish \
  --metrics total \
  --xaxis epoch --smooth 1000 --out_png plots/tvf_swish_loss_compare.png \
  --yscale log \
