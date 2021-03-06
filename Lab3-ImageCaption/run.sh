#
# show attend tell
#

# train
# python train.py --id showAttendTell --caption_model show_attend_tell --input_json data/cocotalk.json \
# --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att \
# --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-4 \
# --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_showAttendTell \
# --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 5

# eval
# echo "eval show attend tell"
python2 eval.py --model ./log_showAttendTell/model-best.pth \
--infos_path ./log_showAttendTell/infos_showAttendTell-best.pkl \
--image_folder ./eval_img --num_images -1

# # eval karpathy's test split
# echo "karpathy's test split"
# python2 eval.py --num_images 50 \
# --model ./log_showAttendTell/model-best.pth \
# --infos_path ./log_showAttendTell/infos_showAttendTell-best.pkl \
# --language_eval 1 --image_folder
#
# topdown
#

# python train.py --id topDown --caption_model topdown --input_json data/cocotalk.json \
# --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att \
# --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-4 \
# --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_topDown \
# --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 5
