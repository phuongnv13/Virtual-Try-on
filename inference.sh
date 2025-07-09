#### paired
CUDA_VISIBLE_DEVICES=4 python inference.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 1 \
 --model_load_path ./VITONHD_PBE_pose.ckpt \
 --save_dir ./inference_result

#### unpaired
CUDA_VISIBLE_DEVICES=4 python inference.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 4 \
 --model_load_path <model weight path> \
 --unpair \
 --save_dir <save directory>

#### paired repaint
CUDA_VISIBLE_DEVICES=4 python inference.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 4 \
 --model_load_path <model weight path>t \
 --repaint \
 --save_dir <save directory>

#### unpaired repaint
CUDA_VISIBLE_DEVICES=4 python inference.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 4 \
 --model_load_path <model weight path> \
 --unpair \
 --repaint \
 --save_dir <save directory>