import torch


def validate_tensor_shape_2d_4d(t):
    shape = t.shape
    if len(shape) not in (2, 4):
        raise ValueError(
            "Only 2D and 4D tensor shapes are supported. Found "
            "Found tensor of shape {} with {} dims".format(shape, len(shape))
        )


def pad_inner_dims(t, pad_to):
    """ return padded-to-block tensor """
    inner_flattened = t.view(t.shape[0], -1)
    co, inners = inner_flattened.shape
    pad_required = pad_to > 1 and inners % pad_to != 0
    pad_size = pad_to - inners % pad_to if pad_required else 0
    pad = torch.zeros(co, pad_size).to(inner_flattened.data)
    t_padded = torch.cat((inner_flattened, pad), 1)
    return t_padded


def clip_padding(t, orig_shape):
    """ return tensor with clipped padding """
    co = orig_shape[0]
    inners = 1
    for s in orig_shape[1:]:
        inners *= s
    t_clipped = t.view(co, -1)[:, :inners]
    return t_clipped


def permute_to_nhwc(t):
    """ for 4D tensors, convert data layout from NCHW to NHWC """
    res = t.permute(0, 2, 3, 1).contiguous() if t.dim() == 4 else t
    return res


def permute_to_nchw(t):
    """ for 4D tensors, convert data layout from NHWC to NCHW """
    res = t.permute(0, 3, 1, 2).contiguous() if t.dim() == 4 else t
    return res
