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
    ./experiments/celeb_tv0_leakyrelu_250901_101200/logs/val.log \
    > ./plots/leaky_relu_val.log #standard leaky
cat \
    ./experiments/celeb_tv0_leakyrelu_250901_101200/logs/train.log \
    > ./plots/leaky_relu_train.log #standard leaky
cat \
    ./experiments/celeb_tv0_relu_250901_100937/logs/val.log \
    > ./plots/relu_val.log #standard relu
cat \
    ./experiments/celeb_tv0_relu_250901_100937/logs/train.log \
    > ./plots/relu_train.log #standard relu
    

#### Curves of metrics
# Compare three runs; full grid (epoch & iteration), mark best, smooth over 5 points
python plot_evaluation.py ./plots/tv0_relu.log ./plots/tv0_leaky_relu.log ./plots/tv0.log --labels relu leaky_relu swish --smooth 5 --output plots/relu_leaky_relu_swish_comp.png --epoch-only
python plot_evaluation.py ./plots/wave_val.log ./plots/tvf_val.log ./plots/tv0.log --labels wave_swish tvf_swish swish --smooth 5 --output plots/wave_tvf_swish_comp.png --epoch-only
python plot_evaluation.py ./plots/tv0_relu.log ./plots/relu_val.log ./plots/tv0.log --labels stdrelu relu swish --smooth 5 --output plots/std_relu_comp.png --epoch-only
python plot_evaluation.py ./plots/tv0_leaky_relu.log ./plots/leaky_relu_val.log ./plots/tv0.log --labels stdleakyrelu leakyrelu swish --smooth 5 --output plots/std_leaky_relu_comp.png --epoch-only

# Dual-y single chart (epoch vs PSNR/SSIM together)
python plot_evaluation.py ./plots/tv0_relu.log ./plots/tv0.log --dual-y --output plots/relu_swish.png --metrics psnr --smooth 5 --label relu swish --max-epoch 140

python plot_evaluation.py  ./plots/wave_val.log ./plots/tvf_val.log ./plots/tv0.log --dual-y --output plots/wave_tvf_swish.png --metrics psnr --smooth 5 --label wave_swish tvf_swish swish --max-epoch 140
# Also dump parsed points to CSV
python plot_evaluation.py ./plots/tv0_relu.log --out-csv plots/relu.csv
python plot_evaluation.py ./plots/tv0_leaky_relu.log --out-csv plots/leaky_relu.csv
python plot_evaluation.py ./plots/tv0.log --out-csv plots/swish.csv
python plot_evaluation.py ./plots/tvf_val.log --out-csv plots/tvf_swish.csv
python plot_evaluation.py ./plots/wave_val.log --out-csv plots/wave_swish.csv

#### Images of evaluations
# Default (recommended): compare the "latest"/max leadnum across folders, per idx
python training_eval_images.py ./experiments/celeb_tv0_leakystdrelu_250825_150651/results ./experiments/celeb_tv0_stdleakyrelu_250827_230725/results  --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix leaky_relu_ --separate-groups
python training_eval_images.py ./experiments/celeb_tv0_stdrelu_250824_162052/results ./experiments/celeb_tv0_stdrelu_250825_151239/results --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix relu_ --separate-groups
python training_eval_images.py ./experiments/celeb_tv0_250827_231027/results --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix swish_ --separate-groups
python training_eval_images.py ./experiments/celeb_tv0_tvf_250829_165656/results  --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix tvf_swish_ --separate-groups

python training_eval_images.py ./experiments/celeb_tv0_wavelet_250830_105558/results  --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix wave_swish_ --separate-groups

python training_eval_images.py ./experiments/celeb_tv0_leakyrelu_250901_101200/results  --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix standard_leakyrelu_ --separate-groups

python training_eval_images.py ./experiments/celeb_tv0_relu_250901_100937/results  --min-subdir 2 --max-subdir 81 --output-dir  plots --per-row 10 --prefix standard_relu_ --separate-groups

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
  --labels tvf_swish \
  --smooth 1000 --out_png plots/tvf_loss_curves.png \
  --xaxis epoch \
#   --yscale log
python plot_losses.py  plots/wave_train.log \
  --metrics total loss_noise loss_wave_l1 \
  --labels wave_swish \
  --smooth 1000 --out_png plots/wave_loss_curves.png \
  --xaxis epoch \


# compare two logs
python plot_losses.py plots/tv0_relu_train.log plots/tv0_leaky_relu_train.log plots/tv0_train.log\
  --labels relu leaky_relu swish \
  --metrics total \
  --xaxis epoch --smooth 1000 --out_png plots/loss_compare_swish_leaky_relu.png \
  --yscale log \
#   --iter-min 20000 --iter-max 550000


python plot_losses.py plots/wave_train.log plots/tvf_train.log  plots/tv0_train.log \
  --labels wave_swish tvf_swish swish \
  --metrics loss_noise \
  --xaxis epoch --smooth 1000 --out_png plots/wave_tvf_swish_loss_compare.png \
  --yscale log \

python plot_losses.py plots/relu_train.log plots/tv0_relu_train.log  plots/tv0_train.log \
  --labels standar_relu std_relu swish \
  --metrics loss_noise \
  --xaxis epoch --smooth 1000 --out_png plots/relu_loss_compare.png \
  --yscale log \

python plot_losses.py plots/leaky_relu_train.log plots/tv0_leaky_relu_train.log  plots/tv0_train.log \
  --labels standar_leaky_relu std_leaky_relu swish \
  --metrics loss_noise \
  --xaxis epoch --smooth 1000 --out_png plots/leaky_relu_loss_compare.png \
  --yscale log \