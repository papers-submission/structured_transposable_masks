git clone https://github.com/NM-sparsity/NM-sparsity.git
cd NM-sparsity
git checkout d8419d99ad84ae47e3581db0125ed375ee416bb3
cd ..
cp src/dist_utils.py NM-sparsity/devkit/core/
cp src/sparse_ops.py NM-sparsity/devkit/sparse_ops/
cp src/train_imagenet.py NM-sparsity/classification/train_imagenet.py
cp src/resnet.py NM-sparsity/classification/models/
cp src/train_val.sh NM-sparsity/classification
cp src/sparse_ops_init.py NM-sparsity/devkit/sparse_ops/__init__.py
cp src/utils.py NM-sparsity/devkit/core/
