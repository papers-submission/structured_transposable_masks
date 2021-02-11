import os
import shutil
import argparse
from torch import load as torch_load, save as torch_save, ones_like as torch_ones_like
from common.timer import Timer
from prune.pruning_method_utils import permute_to_nhwc, pad_inner_dims
from prune.pruning_method_transposable_block_l1 import PruningMethodTransposableBlockL1


def get_args():
    parser = argparse.ArgumentParser(description='Pruner')
    parser.add_argument('--checkpoint', required=True, type=str, help='path to checkpoint')
    parser.add_argument('--n-workers', default=None, type=int, help='number of processes')
    parser.add_argument('--save', default=None, type=str, help='path to pruned checkpoint')
    parser.add_argument('--bs', default=8, type=int, help='block size')
    parser.add_argument('--topk', default=4, type=int, help='topk')
    parser.add_argument('--sd-key', default='state_dict', type=str, help='state dict key in checkpoint')
    parser.add_argument('--optimize-transposed', action='store_true', default=False,
                        help='if true, transposable pruning method will optimize for (block + block.T)')
    parser.add_argument('--include', nargs='*', default=None,
                        help='list of layers that will be included in pruning')
    parser.add_argument('--exclude', nargs='*', default=None,
                        help='list of layers that will be excluded from pruning')
    parser.add_argument('--debug-key', default=None, type=str, help='variable key to print first block')
    args = parser.parse_args()
    return args


def load_checkpoint(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError('Checkpoint {} not found'.format(filename))

    checkpoint = torch_load(filename, map_location='cpu')
    return checkpoint


def load_sd_from_checkpoint(filename, sd_key):
    checkpoint = load_checkpoint(filename)
    sd = checkpoint[sd_key] if sd_key is not None else checkpoint.copy()
    del checkpoint
    return sd


def load_var_from_checkpoint(filename, name, sd_key):
    sd = load_sd_from_checkpoint(filename, sd_key)
    if name not in sd:
        raise RuntimeError('Variable {} not found in {}'.format(name, filename))
    v = sd[name]
    del sd
    return v


def save_var_to_checkpoint(filename, name, mask, sd_key):
    checkpoint = load_checkpoint(filename)
    sd = checkpoint[sd_key] if sd_key is not None else checkpoint
    if name not in sd:
        raise RuntimeError('Variable {} not found in {}'.format(name, filename))
    sd[name] = sd[name] * mask
    torch_save(checkpoint, filename)
    del checkpoint


def prune(checkpoint, save, sd_key, bs=8, topk=4, optimize_transposed=False,
          include=None, exclude=None, n_workers=None, debug_key=None):

    with Timer() as t:
        sd = load_sd_from_checkpoint(checkpoint, sd_key)
    print('Loading checkpoint, elapsed={}'.format(t.total()))

    save = checkpoint + '.pruned' if save is None else save
    shutil.copyfile(checkpoint, save)

    prune_method = PruningMethodTransposableBlockL1(block_size=bs, topk=topk,
                                                    optimize_transposed=optimize_transposed,
                                                    n_workers=n_workers, with_tqdm=True)

    keys = [k for k in sd.keys() if sd[k].dim() > 1 and 'bias' not in k and 'running' not in k]

    if include:
        invalid_keys = [k for k in include if k not in keys]
        assert not invalid_keys, 'Requested params to include={} not in model'.format(invalid_keys)
        print('Including {}'.format(exclude))
        keys = include

    if exclude:
        invalid_keys = [k for k in exclude if k not in keys]
        assert not invalid_keys, 'Requested params to exclude={} not in model'.format(invalid_keys)
        print('Excluding {}'.format(exclude))
        keys = [k for k in keys if k not in exclude]

    del sd

    with Timer() as t:
        for key in keys:
            v = load_var_from_checkpoint(checkpoint, key, sd_key)
            print('Pruning ' + key)
            prune_weight_mask = prune_method.compute_mask(v, torch_ones_like(v))
            save_var_to_checkpoint(save, key, prune_weight_mask, sd_key)
    print('Total elapsed time: {}'.format(t.total()))

    if debug_key:
        bs = bs
        sd = load_sd_from_checkpoint(save, sd_key)
        v = sd[debug_key]

        # print first block
        permuted_mask = permute_to_nhwc(v)
        permuted_mask = pad_inner_dims(permuted_mask, bs * bs)
        permuted_mask = permuted_mask.reshape(-1, (bs * bs))
        print('first block=\n{}'.format(permuted_mask.numpy()[0, :].reshape(1, -1, bs, bs)))


def main():
    args = get_args()
    prune(checkpoint=args.checkpoint, save=args.save, sd_key=args.sd_key, bs=args.bs, topk=args.topk,
          optimize_transposed=args.optimize_transposed, include=args.include, exclude=args.exclude,
          n_workers=args.n_workers, debug_key=args.debug_key)


if __name__ == '__main__':
    main()
