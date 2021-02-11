export datasets_dir=/home/Datasets
export model=${1:-"resnet"}
export model_vis=${2:-"resnet50"}
export depth=${3:-50}
export adaprune_suffix=''
if [ "$5" = True ]; then
    export adaprune_suffix='.adaprune'
fi
export workdir='dense_'${model_vis}$adaprune_suffix
export perC=True


echo ./results/$workdir/resnet
#Download and absorb_bn resnet50 and
python main.py --model $model --save $workdir -b 128  -lfv $model_vis --model-config "{'batch_norm': False,'depth':$depth}" --device-id 1

# Run adaprune to minimize MSE of the output with respect to a perturations in parameters
python main.py --optimize-weights  --model $model -b 200 --evaluate results/$workdir/$model.absorb_bn --model-config "{'batch_norm': False,'depth':$depth}" --dataset imagenet_calib --datasets-dir $datasets_dir --adaprune --prune_bs 8 --prune_topk 4 --device-id 0 --keep_first_last #--unstructured --sparsity_level 0.5
python main.py --batch-norn-tuning --model $model -lfv $model_vis -b 200 --evaluate results/$workdir/$model.absorb_bn.adaprune --model-config "{'batch_norm': False,'depth':$depth}" --dataset imagenet_calib --datasets-dir $datasets_dir --device-id 0

