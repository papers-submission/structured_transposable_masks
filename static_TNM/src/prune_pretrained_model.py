import os
import torch
import argparse
import torchvision.models as models
from torch.hub import get_dir
from glob import glob
from prune.prune import prune


def main():
    # get supported models
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
    parser.add_argument('--save', default=None, type=str, help='pruned checkpoint')
    args = parser.parse_args()

    # if required, download pre trained model
    models.__dict__[args.arch](pretrained=True)

    # get pre trained checkpoint
    checkpoint_path = os.path.join(get_dir(), 'checkpoints')
    files = glob(os.path.join(checkpoint_path, '{}-*.pth').format(args.arch))
    assert len(files) == 1
    checkpoint_file = files[0]

    # prune and save checkpoint
    prune(checkpoint=checkpoint_file, save=args.save, sd_key=None, bs=8, topk=4)

    # add expected fields to checkpoint
    sd = torch.load(args.save)
    checkpoint = {'state_dict': sd, 'epoch': 0, 'best_prec1': 0}
    torch.save(checkpoint, args.save)


if __name__ == '__main__':
    main()
