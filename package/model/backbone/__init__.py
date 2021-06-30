import copy
from .resnet import ResNet
from .ghostnet import GhostNet
from .shufflenetv2 import ShuffleNetV2
from .ic_shufflenetv2 import Ic_ShuffleNetV2
from .mobilenetv2 import MobileNetV2
from .efficientnet_lite import EfficientNetLite
from .custom_csp import CustomCspNet
from .repvgg import RepVGG
from .ic_resnet import Ic_resnet50
pattern_path = 'package/model/backbone/ic_resnet50_k9.json'

def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop('name')
    if name == 'ResNet':
        return ResNet(**backbone_cfg)
    elif name == 'ShuffleNetV2':
        return ShuffleNetV2(**backbone_cfg)
    elif name == 'Ic_ShuffleNetV2':
        return Ic_ShuffleNetV2(**backbone_cfg)
    elif name == 'Ic_resnet50':
        return Ic_resnet50(**backbone_cfg)
    elif name == 'GhostNet':
        return GhostNet(**backbone_cfg)
    elif name == 'MobileNetV2':
        return MobileNetV2(**backbone_cfg)
    elif name == 'EfficientNetLite':
        return EfficientNetLite(**backbone_cfg)
    elif name == 'CustomCspNet':
        return CustomCspNet(**backbone_cfg)
    elif name == 'RepVGG':
        return RepVGG(**backbone_cfg)
    else:
        raise NotImplementedError

