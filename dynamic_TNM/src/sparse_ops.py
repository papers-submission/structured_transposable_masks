import torch
from torch import autograd, nn
import torch.nn.functional as F

from itertools import repeat
from torch._six import container_abcs
import time
from prune.pruning_method_transposable_block_l1 import PruningMethodTransposableBlockL1

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

def update_mask_approx2(data, mask, topk=4,BS=8):
    mask.fill_(0)
    Co = data.shape[0]
    #topk=BS//2
    _,idx_sort = data.sort(1,descending=True); #block x 64
    for k in range(BS**2):
        if k<topk:
           mask[range(Co),idx_sort[range(Co),k]] = 1
        else:
            ii,jj = idx_sort//BS,idx_sort%BS
            row_cond= mask.view(-1,BS,BS).sum(1)[range(Co),ii[:,k]]<topk
            col_cond = mask.view(-1,BS,BS).sum(2)[range(Co),jj[:,k]]<topk
            if (~row_cond).all() and (~col_cond).all():
                break
            idx_sort[row_cond.mul(col_cond)][:,k]
            mask[row_cond.mul(col_cond),idx_sort[row_cond.mul(col_cond)][:,k]]=1
    return mask
    
def update_mask(data, mask, BS=8, top_k_eval=32, max_steps=1):
    Co = data.shape[0]
    for k in range(max_steps):
        val_max, ind_max = data.mul(1 - mask).topk(top_k_eval, 1, sorted=False)
        iii, jjj = ind_max // BS, ind_max % BS

        copy_data = data.clone()
        copy_data[mask == 0] = 1e9

        mpc = copy_data.reshape(-1, BS, BS).min(1)[1]
        mpr = copy_data.reshape(-1, BS, BS).min(2)[1]
        out_r = mpr.gather(1, iii)
        out_c = mpc.gather(1, jjj)

        ind_out = torch.cat([iii * BS + out_r, out_c * BS + jjj]).reshape(2, Co, top_k_eval)
        ind_in_new = out_c * BS + out_r
        ind_in = torch.cat([ind_max, ind_in_new]).reshape(2, Co, top_k_eval)
        val_in = data.mul(1 - mask).gather(1, ind_in[1])
        val_min1 = data.gather(1, ind_out[0])
        val_min2 = data.gather(1, ind_out[1])

        mask_change_val, mask_change_ind = (val_max + val_in - val_min1 - val_min2).max(1)

        ind_in = ind_in[:, range(Co), mask_change_ind].t()
        ind_out = ind_out[:, range(Co), mask_change_ind].t()
        block_masks_in = torch.zeros_like(data).reshape(Co, -1).scatter_(1, ind_in, torch.ones_like(ind_in).float())
        block_masks_out = torch.zeros_like(data).reshape(Co, -1).scatter_(1, ind_out, torch.ones_like(ind_out).float())
        new_mask = (mask + block_masks_in - block_masks_out).clamp(0, 1)

        mask_update = mask_change_val > 0
        mask[mask_update] = new_mask[mask_update]
        if sum(mask_update) == 0:
            break
    return mask

class SparseTranspose(autograd.Function):
    """" Prune the unimprotant edges for the forwards phase but pass the gradient to dense weight using STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, counter, freq, absorb_mean):
        weight.mask = weight.mask.to(weight)
        output = weight.clone()
        if counter%freq==0:
            weight_temp = weight.detach().abs().reshape(-1, M*M)
            weight_mask = weight.mask.detach().reshape(-1, M*M)
            #weight_mask = update_mask(weight_temp,weight_mask,BS=M)
            weight_mask = update_mask_approx2(weight_temp,weight_mask,BS=M)
            if absorb_mean:
                output = output.reshape(-1, M*M).clone()
                output+=output.mul(1-weight_mask).mean(1)
                output=output.reshape(weight.shape)
            weight.mask=weight_mask.reshape(weight.shape)
        return output*weight.mask, weight.mask

    @staticmethod
    def backward(ctx, grad_output, _):
        return grad_output, None, None, None, None, None


class Sparse(autograd.Function):
    """" Prune the unimprotant edges for the forwards phase but pass the gradient to dense weight using STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M):

        output = weight.clone()
        length = weight.numel()
        group = int(length/M)

        weight_temp = weight.detach().abs().reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)

        return output*w_b, w_b


    @staticmethod
    def backward(ctx, grad_output, _):
        return grad_output, None, None

class SparseTransposeV2(autograd.Function):
    """" Prune the unimprotant edges for the forwards phase but pass the gradient to dense weight using STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, counter):
        weight.mask = weight.mask.to(weight)
        output = weight.reshape(-1, M*M).clone()
        weight_mask = weight.mask.reshape(-1, M*M)
        output+=torch.mean(output.mul(1-weight_mask),dim=1,keepdim=True)
        weight.mask=weight_mask.reshape(weight.shape)
        output=output.reshape(weight.shape)
        return output*weight.mask, weight.mask

    @staticmethod
    def backward(ctx, grad_output, _):
        return grad_output, None, None, None

class SparseConvTranspose(nn.Conv2d):
    

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', N=2, M=4, **kwargs):
        self.N = N
        self.M = M
        self.counter = 0
        self.freq = 1
        self.absorb_mean = False
        super(SparseConvTranspose, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **kwargs)


    def get_sparse_weights(self):
        return SparseTranspose.apply(self.weight, self.N, self.M, self.counter, self.freq, self.absorb_mean)



    def forward(self, x):
        if self.training:
            self.counter+=1
            self.freq = 40 #min(self.freq+self.counter//100,100)
            w, mask = self.get_sparse_weights()
            setattr(self.weight, "mask", mask)
        else:
            w = self.weight * self.weight.mask
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SparseConvTransposeV2(nn.Conv2d):
    

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', N=2, M=4, **kwargs):
        self.N = N
        self.M = M
        self.counter = 0
        self.freq = 1
        self.rerun_ip = 0.01
        self.ipClass = PruningMethodTransposableBlockL1(block_size=self.M, topk=self.N)
        super(SparseConvTransposeV2, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **kwargs)


    def get_sparse_weights(self):
        with torch.no_grad():
            weight_temp = self.weight.detach().abs().reshape(-1, self.M*self.M)
            weight_mask = self.weight.mask.detach().reshape(-1, self.M*self.M)
            num_samples_ip= int(self.rerun_ip*weight_temp.shape[0])
            idx=torch.randperm(weight_temp.shape[0])[:num_samples_ip]
            sample_weight = weight_temp[idx]
            mask_new = self.ipClass.compute_mask(sample_weight,torch.ones_like(sample_weight))
            weight_mask = weight_mask.to(self.weight.device)
            weight_mask[idx]=mask_new.to(self.weight.device)
        return SparseTransposeV2.apply(self.weight, self.N, self.M, self.counter)

    def forward(self, x):
        # self.counter+=1
        # self.freq = min(self.freq+self.counter//100,100)
        w, mask = self.get_sparse_weights()
        setattr(self.weight, "mask", mask)
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SparseConv(nn.Conv2d):
    

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', N=2, M=4, **kwargs):
        self.N = N
        self.M = M
        super(SparseConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **kwargs)


    def get_sparse_weights(self):

        return Sparse.apply(self.weight, self.N, self.M)



    def forward(self, x):

        w, mask = self.get_sparse_weights()
        setattr(self.weight, "mask", mask)
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SparseLinear(nn.Linear):
    def __init__():

        self.N = N
        self.M = M

    


class SparseLinearTranspose(nn.Linear):

    def __init__(self, in_channels, out_channels,  bias=True, N=2, M=4,  **kwargs):
        self.N = N
        self.M = M
        self.counter = 0
        self.freq = 10
        super(SparseLinearTranspose, self).__init__(in_channels, out_channels, bias,)

    def get_sparse_weights(self):
        return SparseTranspose.apply(self.weight, self.N, self.M, self.counter, self.freq, False)

    def forward(self, x):
        if self.training:
            self.counter += 1
            self.freq = 40  # min(self.freq+self.counter//100,100)
            w, mask = self.get_sparse_weights()
            setattr(self.weight, "mask", mask)
        else:
            w = self.weight * self.weight.mask
        x = F.linear(
            x, w, self.bias
        )
        return x
