import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import scipy.optimize as opt
import math

        

def adaprune(layer, mask, cached_inps, cached_outs, test_inp, test_out, lr1=1e-4, lr2=1e-2, iters=1000, progress=True, batch_size=50,relu=False,bs=8,no_optimization=False,keep_first_last=True):
    print("\nRun adaprune")
    test_inp = test_inp.to(layer.weight.device)
    test_out = test_out.to(layer.weight.device)
    layer.quantize=False
    if keep_first_last and (layer.weight.dim()==2 or layer.weight.shape[1]==3):
        return 0.1, 0.1    
    with torch.no_grad():
        layer.weight.data = absorb_mean_to_nz(layer.weight,mask,bs=bs)
        layer.weight.mul_(mask.to(layer.weight.device))    
    mse_before = F.mse_loss(layer(test_inp), test_out)
    if no_optimization:
        return mse_before.item(),mse_before.item()

    lr_w = 1e-3
    lr_b = 1e-2

    opt_w = torch.optim.Adam([layer.weight], lr=lr_w)
    if hasattr(layer, 'bias') and layer.bias is not None: opt_bias = torch.optim.Adam([layer.bias], lr=lr_b)

    losses = []

    for j in (tqdm(range(iters)) if progress else range(iters)):
        idx = torch.randperm(cached_inps.size(0))[:batch_size]

        train_inp = cached_inps[idx].to(layer.weight.device)
        train_out = cached_outs[idx].to(layer.weight.device)
        qout = layer(train_inp)
        if relu:
            loss = F.mse_loss(F.relu(qout), F.relu(train_out))
        else:    
            loss = F.mse_loss(qout, train_out)
        
        losses.append(loss.item())
        opt_w.zero_grad()
        if hasattr(layer, 'bias') and layer.bias is not None: opt_bias.zero_grad()
        loss.backward()
        opt_w.step()
        if hasattr(layer, 'bias') and layer.bias is not None: opt_bias.step()
        with torch.no_grad():
            layer.weight.mul_(mask.to(layer.weight.device))

    mse_after = F.mse_loss(layer(test_inp), test_out)
    return mse_before.item(), mse_after.item()

def absorb_mean_to_nz(weight,mask,bs=8):
    """Prunes the weights with smallest magnitude."""
    if weight.dim()>2:
        Co,Ci,k1,k2=weight.shape
        pad_size=bs-(Ci*k1*k2)%bs if bs>1 else 0
        weight_pad = torch.cat((weight.permute(0,2,3,1).contiguous().view(Co,-1),torch.zeros(Co,pad_size).to(weight.data)),1)
        mask_pad = torch.cat((mask.permute(0,2,3,1).contiguous().view(Co,-1).float(),torch.ones(Co,pad_size).to(weight.data).float()),1)
    else:        
        Co,Ci=weight.shape
        pad_size=bs-Ci%bs if bs>1 else 0
        weight_pad = torch.cat((weight.view(Co,-1),torch.zeros(Co,pad_size).to(weight.data)),1)
        mask_pad = torch.cat((mask.view(Co,-1).float(),torch.ones(Co,pad_size).to(weight.data).float()),1)
    
    weight_pad = weight_pad.view(Co,-1,bs)+weight_pad.view(Co,-1,bs).mul(1-mask_pad.view(Co,-1,bs)).sum(2,keepdim=True).div(mask_pad.view(Co,-1,bs).sum(2,keepdim=True))
    weight_pad.mul_(mask_pad.view(Co,-1,bs))
    if weight.dim()>2:
           weight_pad = weight_pad.view(Co,-1)[:,:Ci*k1*k2]
           weight_pad = weight_pad.view(Co,k1,k2,Ci).permute(0,3,1,2) 
    else:        
        weight_pad = weight_pad.view(Co,-1)[:,:Ci]
    return weight_pad

def create_block_magnitude_mask(weight, bs=2, topk=1):
        """Prunes the weights with smallest magnitude."""
        if weight.dim()>2:
            Co,Ci,k1,k2=weight.shape
            pad_size=bs-(Ci*k1*k2)%bs if bs>1 else 0
            weight_pad = torch.cat((weight.permute(0,2,3,1).contiguous().view(Co,-1),torch.zeros(Co,pad_size).to(weight.data)),1)
        else:        
            Co,Ci=weight.shape
            pad_size=bs-Ci%bs if bs>1 else 0
            weight_pad = torch.cat((weight.view(Co,-1),torch.zeros(Co,pad_size).to(weight.data)),1)
    
        block_weight = weight_pad.data.abs().view(Co,-1,bs).topk(k=topk,dim=2,sorted=False)[1].reshape(Co,-1,topk)
        block_masks = torch.zeros_like(weight_pad).reshape(Co, -1, bs).scatter_(2, block_weight, torch.ones(block_weight.shape).to(weight))

        if weight.dim()>2:
            block_masks = block_masks.view(Co,-1)[:,:Ci*k1*k2]
            block_masks = block_masks.view(Co,k1,k2,Ci).permute(0,3,1,2) 
        else:        
            block_masks = block_masks.view(Co,-1)[:,:Ci]
        return block_masks

def create_global_unstructured_magnitude_mask(param,global_val):
    eps = 0.1 if param.shape[1]==3 else 0
    return param.abs().gt(global_val-eps)

def create_unstructured_magnitude_mask(param,sparsity_level,absorb_mean=True):
    topk = int(param.numel()*sparsity_level)
    val = param.view(-1).abs().topk(topk,sorted=True)[0][-1]
    mask = param.abs().gt(val)
    if absorb_mean:
        with torch.no_grad():
            mean_val=param[~mask].mean()
            aa = param+mask*mean_val
            param.copy_(aa)  
    print('unstructured mask created with %f sparsity'%(mask.sum().float()/mask.numel()))
    return mask

def extract_topk(param,bs,global_val,conf_level=0.95):
    if global_val is not None:
        param = create_global_unstructured_magnitude_mask(param,global_val)
    p = (1 - param.ne(0).float().sum() / param.numel()).item()
    n = bs
    P=[]
    B=param.numel()/n
    for k in range(n): 
        S = 0
        for i in range(k,n+1):
            C = math.factorial(n)/(math.factorial(i)*math.factorial(n-i))
            S = min(S + C*(p**i)*(1-p)**(n-i),1.0) 
        P.append(S)
    RSD = [math.sqrt((1-pp)/(B*pp)) for pp in P]   
    P_RSD = np.array(P) #- np.array(RSD)*5
    aa = [i for i,p in enumerate(P_RSD) if p>conf_level]
    if len(aa)>0:
        topk = n-[i for i,p in enumerate(P_RSD) if p>conf_level][-1] 
    else:
        topk=n    
    return topk  

def create_mask(layer,bs=8,topk=4,prune_extract_topk=False,unstructured =True,sparsity_level=0.5,global_val=None,conf_level=0.95):
    if unstructured:
        if global_val is not None and not prune_extract_topk:
            print('Creating unstructured mask for layer %s'%(layer.name))
            return create_global_unstructured_magnitude_mask(layer.weight,global_val,conf_level)
        else:
            return create_unstructured_magnitude_mask(layer.weight,sparsity_level=sparsity_level) 
    if prune_extract_topk: topk = extract_topk(layer.weight,bs,global_val,conf_level=conf_level)
    print('Creating mask for layer %s with bs %d ,topk %d'%(layer.name,bs,topk))
    return create_block_magnitude_mask(layer.weight,bs=bs,topk=topk)

def optimize_layer(layer, in_out, optimize_weights=False,bs=4,topk=2,extract_topk=False,unstructured=False,sparsity_level=0.5,global_val=None,conf_level=0.95):
    batch_size = 100

    cached_inps = torch.cat([x[0] for x in in_out])
    cached_outs = torch.cat([x[1] for x in in_out])

    idx = torch.randperm(cached_inps.size(0))[:batch_size]

    test_inp = cached_inps[idx]
    test_out = cached_outs[idx]

    if optimize_weights:
        mask = create_mask(layer,bs=bs,topk=topk,prune_extract_topk=extract_topk,unstructured=unstructured,sparsity_level=sparsity_level,global_val=global_val,conf_level=conf_level)
        if 'conv1' in layer.name or 'conv2' in layer.name:
            mse_before, mse_after = adaprune(layer, mask, cached_inps, cached_outs, test_inp, test_out, iters=1000, lr1=1e-5, lr2=1e-4,relu=False,bs=bs)
        else:
            mse_before, mse_after = adaprune(layer, mask, cached_inps, cached_outs, test_inp, test_out, iters=1000, lr1=1e-5, lr2=1e-4,relu=False,bs=bs)    

        mse_before_opt = mse_before
        print("MSE before adaprune (opt weight): {}".format(mse_before))
        print("MSE after adaprune (opt weight): {}".format(mse_after))
        torch.cuda.empty_cache()
    else:
        mse_before, mse_after = optimize_qparams(layer, cached_inps, cached_outs, test_inp, test_out)
        mse_before_opt = mse_before
        print("MSE before qparams: {}".format(mse_before))
        print("MSE after qparams: {}".format(mse_after))

    mse_after_opt = mse_after

    with torch.no_grad():
        N = test_out.numel()
        snr_before = (1/math.sqrt(N)) * math.sqrt(N * mse_before_opt) / torch.norm(test_out).item()
        snr_after = (1/math.sqrt(N)) * math.sqrt(N * mse_after_opt) / torch.norm(test_out).item()


    kurt_in = kurtosis(test_inp).item()
    kurt_w = kurtosis(layer.weight).item()

    del cached_inps
    del cached_outs
    torch.cuda.empty_cache()

    return mse_before_opt, mse_after_opt, snr_before, snr_after, kurt_in, kurt_w, mask


def kurtosis(x):
    var = torch.mean((x - x.mean())**2)
    return torch.mean((x - x.mean())**4 / var**2)


def dump(model_name, layer, in_out):
    path = os.path.join("dump", model_name, layer.name)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    if hasattr(layer, 'groups'):
        f = open(os.path.join(path, "groups_{}".format(layer.groups)), 'x')
        f.close()

    cached_inps = torch.cat([x[0] for x in in_out])
    cached_outs = torch.cat([x[1] for x in in_out])
    torch.save(cached_inps, os.path.join(path, "input.pt"))
    torch.save(cached_outs, os.path.join(path, "output.pt"))
    torch.save(layer.weight, os.path.join(path, 'weight.pt'))
    if layer.bias is not None:
        torch.save(layer.bias, os.path.join(path, 'bias.pt'))


