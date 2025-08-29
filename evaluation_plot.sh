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


#### Curves of metrics 
# Compare three runs; full grid (epoch & iteration), mark best, smooth over 5 points
python plot_evaluation.py ./plots/tv0_relu.log ./plots/tv0_leaky_relu.log ./plots/tv0.log --labels relu leaky_relu swish --smooth 5 --output plots/relu_leaky_relu_swish_comp.png --epoch-only
# Dual-y single chart (epoch vs PSNR/SSIM together)
python plot_evaluation.py ./plots/tv0_relu.log ./plots/tv0.log --dual-y --output plots/relu_swish.png --metrics psnr --smooth 5 --label relu swish --max-epoch 80
# Also dump parsed points to CSV
python plot_evaluation.py ./plots/tv0_relu.log --out-csv plots/relu.csv
python plot_evaluation.py ./plots/tv0_leaky_relu.log --out-csv plots/leaky_relu.csv
python plot_evaluation.py ./plots/tv0.log --out-csv plots/swish.csv

#### Images of evaluations
# Default (recommended): compare the "latest"/max leadnum across folders, per idx
python training_eval_images.py ./experiments/celeb_tv0_leakystdrelu_250825_150651/results ./experiments/celeb_tv0_stdleakyrelu_250827_230725/results  --min-subdir 2 --max-subdir 80 --output-dir  plots --per-row 10 --prefix leaky_relu_ --separate-groups
python training_eval_images.py ./experiments/celeb_tv0_stdrelu_250824_162052/results ./experiments/celeb_tv0_stdrelu_250825_151239/results --min-subdir 2 --max-subdir 80 --output-dir  plots --per-row 10 --prefix relu_ --separate-groups
python training_eval_images.py ./experiments/celeb_tv0_250827_231027/results --min-subdir 2 --max-subdir 80 --output-dir  plots --per-row 10 --prefix swish_ --separate-groups



##### Plot of losses with smoothing
# single log
# metrics:  total loss_noise loss_TV1 loss_TV2 loss_TVF loss_wave_l1
python plot_losses.py  plots/tv0_relu_train.log \
  --metrics total loss_noise loss_TV1 loss_TV2 loss_TVF loss_wave_l1 \
  --labels relu \
  --smooth 1000 --out_png plots/loss_curves.png \
  --xaxis epoch \
#   --yscale log
# compare two logs
python plot_losses.py plots/tv0_relu_train.log plots/tv0_leaky_relu_train.log \
  --labels relu leaky_relu \
  --metrics total \
  --xaxis iter --smooth 1000 --out_png plots/loss_compare.png \
  --yscale log \
  --iter-min 20000 --iter-max 550000


