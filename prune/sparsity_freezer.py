import torch
from .pruning_method_based_mask import PruningMethodBasedMask


class SparsityFreezer:
    """ Keeps sparsity level in model by applying pruning based on current zeros of parameters """
    @staticmethod
    def freeze(model):
        with torch.no_grad():
            params = SparsityFreezer._get_model_params(model)
            SparsityFreezer._enforce_mask_based_on_zeros(params)

    @staticmethod
    def _enforce_mask_based_on_zeros(params):
        prune_method = PruningMethodBasedMask()
        for param_info in params.values():
            module, name = param_info
            param = getattr(module, name)
            mask = param.ne(0).float()
            prune_method.apply_like_self(module=module, name=name, mask=mask)

    @staticmethod
    def _get_model_params(model):
        params = {}
        for m_info in list(model.named_modules()):
            module_name, module = m_info
            for p_info in list(module.named_parameters(recurse=False)):
                param_name, param = p_info
                key = module_name + '.' + param_name
                if param.dim() > 1 and 'bias' not in key and 'running' not in key:
                    params[key] = (module, param_name)

        # a shared parameter will only appear once in model.named_parameters()
        # therefore, filter to get only parameters that appear in model.named_parameters()
        model_named_params = set([name for name, _ in model.named_parameters()])
        params = {p: v for p, v in params.items() if p in model_named_params}
        return params
