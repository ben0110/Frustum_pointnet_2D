#/bin/bash
python train/train.py --gpu 0 --model frustum_pointnets_v2 --log_dir train/log_v2 --num_point 3500 --max_epoch 201 --batch_size 32 --decay_step 800000 --decay_rate 0.5 #--restore_model_path train/log_v2/06-05-2020-20:56:24/ckpt/model_200.ckpt
