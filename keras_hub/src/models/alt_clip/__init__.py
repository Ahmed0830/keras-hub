from keras_hub.src.models.alt_clip.alt_clip_backbone import AltCLIPBackbone
from keras_hub.src.models.alt_clip.alt_clip_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, AltCLIPBackbone)
