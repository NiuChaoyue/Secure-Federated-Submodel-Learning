rm -f *.pyc
rm -f *.csv
rm -rf best*
CUDA_VISIBLE_DEVICES=1 python train_taobao.py >log_central_din.txt 2>&1 &
