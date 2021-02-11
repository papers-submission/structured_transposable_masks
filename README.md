# Improving Post Training Neural Quantization:Layer-wise Calibration and Integer Programming
Recently, researchers proposed pruning deep neural network weights (DNNs) using an $N:M$ fine-grained block sparsity mask. In this mask, for each block of M weights, we have at least N zeros. In contrast to unstructured sparsity, N:M fine-grained block sparsity allows acceleration in actual modern hardware. Previously suggested solutions enabled DNN acceleration at the inference phase. To also allow such acceleration in the training phase, we suggest a novel transposable-fine-grained sparsity mask where the same mask can be used for both forward and backward passes. Our transposable mask ensures that both the weight matrix and its transpose follow the same sparsity pattern; thus the matrix multiplication required for passing the error backward can also be accelerated. We discuss the transposable constraint and devise a new measure for mask constraints, called mask-diversity (MD), which correlates with their expected accuracy. Lastly, we formulate the problem of finding the optimal transposable mask as a minimum-cost-flow problem and suggest a fast linear approximation that can be used when the masks dynamically change while training. Our experiments suggest 2x speed-up with no accuracy degradation over vision and language models. A reference implementation is available in the supplementary material.
## Reproducing the results

This repository is partially based on [convNet.pytorch](https://github.com/eladhoffer/convNet.pytorch) repo.  please ensure that you are using pytorch 1.7+.
Reproducing AdaPrune results 
```bash
cd AdaPrune
sh scripts/adaprune_dense_bnt.sh
sh scripts/adaprune_sparse.sh
```
Reproducing static NM-transposable starting from dense pre-trained model:
```bash
cd static_TNM
sh scripts/prune_pretrained_R50.sh
```
Reproducing dynamic NM-transposable from scratch:
```bash
cd dynamic_TNM
sh scripts/clone_and_copy.sh
sh scripts/run_R18.sh
sh scripts/run_R50.sh
```
