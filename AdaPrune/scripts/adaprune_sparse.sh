export datasets_dir=/home/Datasets
export model=${1:-"resnet"}
export model_vis=${2:-"resnet50"}
export depth=${3:-50}
export adaprune_suffix='.adaprune'

export workdir='sparse_'${model_vis}$adaprune_suffix
mkdir ./results/$workdir
echo ./results/$workdir/resnet

#copy sparse model to workdir
cp ./results/resnet50/model_best.pth.tar ./results/$workdir/resnet

# Run adaprune to minimize MSE of the output with respect to a small perturations in parameters
python main.py --optimize-weights  --model $model -b 200 --evaluate results/$workdir/$model --model-config "{'batch_norm': True,'depth':$depth}" --dataset imagenet_calib --datasets-dir $datasets_dir --adaprune  --prune_bs 4 --prune_topk 2

