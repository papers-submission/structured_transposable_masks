export datasets_dir=/datasets
export dataset=imagenet
export workdir='./results/static_TNM'

echo $workdir
cd ..
python -m static_TNM.src.prune_pretrained_model -a resnet50 --save $workdir/resnet50-pruned.pth
cp $workdir/resnet50-pruned.pth $workdir/resnet50.pth
python -m vision.main --model resnet --resume $workdir/resnet50.pth --save $workdir --sparsity-freezer -b 256 --device-ids 0 1 2 3 --dataset $dataset --datasets-dir $datasets_dir
