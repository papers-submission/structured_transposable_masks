import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool
from common.timer import Timer
from prune.pruning_method_utils import *
import numpy as np
import torch.nn.utils.prune as prune


class PruningMethodTransposableBlockL1Graphs(prune.BasePruningMethod):

    PRUNING_TYPE = 'unstructured'  # pruning type "structured" refers to channels

    RUN_SPEED_TEST = False

    def __init__(self, block_size, topk, optimize_transposed=False, n_workers=None, with_tqdm=True):
        super(PruningMethodTransposableBlockL1Graphs, self).__init__()
        assert topk <= block_size
        assert n_workers is None or n_workers > 0
        self.bs = block_size
        self.topk = topk
        self.optimize_transposed = optimize_transposed
        self.n_workers = n_workers
        self.with_tqdm = with_tqdm
        # used for multiprocess in order to avoid serialize/deserialize tensors etc.
        self.mp_tensor, self.mp_mask = None, None

    def nxGraph(self, data):
        bs = data.shape[0]
        G = nx.DiGraph()
        G.add_node('s', demand=-int(bs ** 2 / 2))
        G.add_node('t', demand=int(bs ** 2 / 2))
        names = []
        for i in range(bs):
            G.add_edge('s', 'row' + str(i), capacity=self.topk, weight=0)
            G.add_edge('col' + str(i), 't', capacity=self.topk, weight=0)
            for j in range(bs):
                G.add_edge('row' + str(i), 'col' + str(j), capacity=1, weight=data[i, j].numpy())
            names.append('row' + str(i))
        dictMinFLow = nx.min_cost_flow(G)
        mask = []
        for w in names:
            mask.append(list(dictMinFLow[w].values()))
        return np.array(mask)

    def get_mask_iter(self, c):
        co, inners = self.mp_tensor.shape
        block_numel = self.bs ** 2
        n_blocks = inners // block_numel
        for j in range(n_blocks):
            offset = j * block_numel
            w_block = self.mp_tensor[c, offset:offset + block_numel].reshape(self.bs, self.bs)
            w_block = w_block + w_block.T if self.optimize_transposed else w_block
            mask_block = self.nxGraph(-1 * w_block).reshape(-1) #max flow to min flow
            self.mp_mask[c, offset:offset + block_numel] = torch.from_numpy(mask_block)

    def get_mask(self, t):
        self.mp_tensor = t
        self.mp_mask = torch.zeros_like(t)

        co, inners = t.shape
        n_blocks = inners // (self.bs ** 2)

        if self.RUN_SPEED_TEST:
            self.RUN_SPEED_TEST = False
            with Timer() as t:
                self.get_mask_iter(0)
            elapsed = t.total().total_seconds()
            print('Single core speed test: blocks={} secs={} block-time={}'.format(n_blocks, elapsed, elapsed/n_blocks))

        p = Pool(self.n_workers)
        n_iterations = co
        bar = tqdm(total=n_iterations, ncols=80) if self.with_tqdm else None
        bar.set_postfix_str('n_processes={}, blocks/iter={}'.format(p._processes, n_blocks)) if self.with_tqdm else None
        block_indexes = range(co)
        for _ in p.imap_unordered(self.get_mask_iter, block_indexes):
            bar.update(1) if self.with_tqdm else None
        bar.close() if self.with_tqdm else None
        p.close()

        return self.mp_mask

    def compute_mask(self, t, default_mask):
        # permute and pad
        validate_tensor_shape_2d_4d(t)
        t_masked = t.clone().detach().mul_(default_mask)
        t_permuted = permute_to_nhwc(t_masked)
        pad_to = self.bs ** 2
        t_padded = pad_inner_dims(t_permuted, pad_to)
        t = t_padded.data.abs().to(t)

        # compute mask
        mask = self.get_mask(t)

        # restore to original shape
        block_mask = clip_padding(mask, t_permuted.shape).reshape(t_permuted.shape)
        block_mask = permute_to_nchw(block_mask)
        return block_mask
