from prune.pruning_method_utils import *
import torch.nn.utils.prune as prune


class PruningMethodBasedMask(prune.BasePruningMethod):
    """ Pruning based on fixed mask """

    PRUNING_TYPE = 'unstructured'  # pruning type "structured" refers to channels

    def __init__(self, mask=None):
        super(PruningMethodBasedMask, self).__init__()
        self.mask = mask

    def compute_mask(self, t, default_mask):
        validate_tensor_shape_2d_4d(t)
        mask = self.mask.detach().mul_(default_mask)
        return mask.byte()

    def apply_like_self(self, module, name, **kwargs):
        assert 'mask' in kwargs
        cls = self.__class__
        return super(PruningMethodBasedMask, cls).apply(module, name, kwargs['mask'])
