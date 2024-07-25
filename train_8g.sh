config=/home/shuang.he/workspace/mmdetection/configs/fcos/dev_fcos_r18_fpn_gn-head_1x_coco_v9.py
log_file=${config}.log

bash /home/shuang.he/workspace/mmdetection/train.sh $config 8 2>&1 | tee ${log_file}