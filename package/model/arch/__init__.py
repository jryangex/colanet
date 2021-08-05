import warnings

from .gfl import GFL


def build_model(model_cfg):
    if model_cfg.arch.name == 'GFL':
        warnings.warn("Model architecture name is changed to 'OneStageDetector'. "
                      "The name 'GFL' is deprecated, please change the model->arch->name "
                      "in your YAML config file to OneStageDetector. ")
        model = GFL(model_cfg.arch.backbone, model_cfg.arch.fpn, model_cfg.arch.head)

    else:
        raise NotImplementedError
    return model
