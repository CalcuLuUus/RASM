python train.py --warmup --train_ps 320 \
--batch_size 4 \
--nepoch 1000 \
--gpu '0' \
--env env_name \
--save_dir ./logs/ \
--train_dir path/to/train/data/ \
--val_dir path/to/val/data/ \
--lr_initial 0.0004 