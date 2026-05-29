from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.alt_clip.alt_clip_backbone import AltCLIPBackbone


@keras_hub_export("keras_hub.layers.AltCLIPImageConverter")
class AltCLIPImageConverter(ImageConverter):
    backbone_cls = AltCLIPBackbone
