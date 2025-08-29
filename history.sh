 cat \
    ./experiments/celeb_tv2_250811_004251/logs/val.log \
    ./experiments/celeb_tv2_250811_094801/logs/val.log \
    ./experiments/celeb_tv2_250811_185935/logs/val.log \
    ./experiments/celeb_tv2_250812_152441/logs/val.log \
    > tv2.log

 cat \
    ./experiments/celeb_tv1_250811_004251/logs/val.log \
    ./experiments/celeb_tv1_250811_094800/logs/val.log \
    ./experiments/celeb_tv1_250811_191107/logs/val.log \
    ./experiments/celeb_tv1_250812_145728/logs/val.log \
    ./experiments/celeb_tv1_250813_094827/logs/val.log \
    > tv1.log
 cat \
    ./experiments/celeb_tv1_01_250811_185104/logs/val.log \
    ./experiments/celeb_tv1_01_250812_144559/logs/val.log \
    > tv1_01.log
 cat \
    ./experiments/celeb_tv1_10_250811_185202/logs/val.log \
    ./experiments/celeb_tv1_10_250812_145040/logs/val.log \
    > tv1_10.log
cat \
    ./experiments/celeb_tv2_10_250811_185301/logs/val.log \
    ./experiments/celeb_tv2_10_250812_152608/logs/val.log \
    > tv2_10.log
cat \
    ./experiments/celeb_tv2_01_250811_185203/logs/val.log \
    ./experiments/celeb_tv2_01_250812_154847/logs/val.log \
    > tv2_01.log
cat \
    ./experiments/celeb_tv0_250812_163244/logs/val.log \
    > tv0.log
python plot_history.py \
    tv2.log tv2_01.log tv2_10.log\
    --epoch-only --output comparison_epoch_tv2.png --max-epoch 60
python plot_history.py \
    tv1.log tv1_01.log tv1_10.log\
    --epoch-only --output comparison_epoch_tv1.png --max-epoch 60
python plot_history.py \
    tv0.log tv1.log tv2.log\
    --epoch-only --output comparison_epoch.png --max-epoch 60
python plot_losses.py ./experiments/celeb_tv0_stdrelu_250824_162052/logs/train.log --out_csv plots/losses.csv --out_png plots/losses.png
