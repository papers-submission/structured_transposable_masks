import torch
import os
import shutil
from devkit.sparse_ops import SparseConvTranspose,SparseLinearTranspose


def save_checkpoint(model_dir, state, is_best):
    epoch = state['epoch']
    path = os.path.join(model_dir, 'model.pth-' + str(epoch))
    torch.save(state, path)
    checkpoint_file = os.path.join(model_dir, 'checkpoint')
    checkpoint = open(checkpoint_file, 'w+')
    checkpoint.write('model_checkpoint_path:%s\n' % path)
    checkpoint.close()
    if is_best:
        shutil.copyfile(path, os.path.join(model_dir, 'model-best.pth'))


def load_state(model_dir, model, optimizer=None):
    if not os.path.exists(model_dir + '/checkpoint'):
        print("=> no checkpoint found at '{}', train from scratch".format(model_dir))
        return 0, 0
    else:
        ckpt = open(model_dir + '/checkpoint')
        model_path = ckpt.readlines()[0].split(':')[1].strip('\n')
        checkpoint = torch.load(model_path,map_location='cuda:{}'.format(torch.cuda.current_device()))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        ckpt_keys = set(checkpoint['state_dict'].keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            print('missing keys from checkpoint {}: {}'.format(model_dir, k))

        print("=> loaded model from checkpoint '{}'".format(model_dir))
        if optimizer != None:
            best_prec1 = 0
            if 'best_prec1' in checkpoint.keys():
                best_prec1 = checkpoint['best_prec1']
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> also loaded optimizer from checkpoint '{}' (epoch {})"
                  .format(model_dir, start_epoch))
            return best_prec1, start_epoch


def load_state_epoch(model_dir, model, epoch):
    model_path = model_dir + '/model.pth-' + str(epoch)
    checkpoint = torch.load(model_path,map_location='cuda:{}'.format(torch.cuda.current_device()))

    model.load_state_dict(checkpoint['state_dict'], strict=False)
    ckpt_keys = set(checkpoint['state_dict'].keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    for k in missing_keys:
        print('missing keys from checkpoint {}: {}'.format(model_dir, k))

    print("=> loaded model from checkpoint '{}'".format(model_dir))


def load_state_ckpt(model_path, model):
    checkpoint = torch.load(model_path, map_location='cuda:{}'.format(torch.cuda.current_device()))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    ckpt_keys = set(checkpoint['state_dict'].keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    for k in missing_keys:
        print('missing keys from checkpoint {}: {}'.format(model_path, k))

    print("=> loaded model from checkpoint '{}'".format(model_path))

def save_masks(model,args):
    masks = {}
    for n, m in model.named_modules():
        if isinstance(m, SparseConvTranspose) or isinstance(m,SparseLinearTranspose):
            masks[n] = m.weight.mask.cpu()
    masks['state_dict'] = model.state_dict()
    torch.save(masks, args.mask_path + args.model + '_' + str(args.N) + '_' + str(args.M))

def load_state_and_masks(model, args):
    masks = torch.load(args.mask_path + args.model + '_' + str(args.N) + '_' + str(args.M))

    #load weights
    model.load_state_dict(masks['state_dict'], strict=False)
    ckpt_keys = set(masks['state_dict'].keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    for k in missing_keys:
        print('missing keys from checkpoint {}'.format( k))

    #load_masks
    for n, m in model.named_modules():
        if isinstance(m, SparseConvTranspose) or isinstance(m,SparseLinearTranspose):

      #      m.maskBuff.data = masks[n]
           setattr(m.weight, "mask", masks[n])


