now=$(date +"%Y%m%d_%H%M%S")
export RANK=0
export WORLD_SIZE=8
export PYTHONPATH="path_to_TNM_repo"
python train_imagenet.py \
--config $1 2>&1|tee train-$now.log




