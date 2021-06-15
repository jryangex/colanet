import copy
from .gfl_head import GFLHead
from .colanet_head import ColanetHead


def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop('name')
    if name == 'GFLHead':
        return GFLHead(**head_cfg)
    elif name == 'ColanetHead':
        return ColanetHead(**head_cfg)
    else:
        raise NotImplementedError