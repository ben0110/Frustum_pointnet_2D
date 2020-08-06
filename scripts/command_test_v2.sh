#/bin/bash
python train/test.py --gpu 0 --num_point 3500 --model frustum_pointnets_v2 --model_path '/root/frustum-pointnets_RSC_2D/train/log_v2/06-05-2020-20:56:24/ckpt/model_200.ckpt' --batch_size 1 --output train/detection_results_v2 --idx_path kitti/image_sets/val.txt
