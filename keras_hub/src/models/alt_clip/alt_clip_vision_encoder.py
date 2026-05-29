from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.clip.clip_vision_encoder import CLIPVisionEncoder


@keras_hub_export("keras_hub.models.AltCLIPVisionEncoder")
class AltCLIPVisionEncoder(CLIPVisionEncoder):
    """AltCLIP vision encoder — identical architecture to CLIPVisionEncoder (ViT).

    This is a ViT-based image encoder used in AltCLIP. The architecture is
    identical to `CLIPVisionEncoder`; this subclass exists to provide the
    correct `backbone_cls` for preset resolution.

    See `CLIPVisionEncoder` for full argument documentation.

    Args:
        patch_size: int. The size of each square patch in the input image.
        hidden_dim: int. The size of the transformer hidden state at the end
            of each transformer layer.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        intermediate_activation: activation function. The activation that
            is used for the first Dense layer in a two-layer feedforward
            network for each transformer. Defaults to `"quick_gelu"`.
        intermediate_output_index: optional int. The index of the intermediate
            output.
        image_shape: tuple. The input shape without the batch size. Defaults to
            `(224, 224, 3)`.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the models computations and weights.

    Example:
    ```python
    encoder = keras_hub.models.AltCLIPVisionEncoder(
        patch_size=14,
        hidden_dim=1024,
        num_layers=24,
        num_heads=16,
        intermediate_dim=4096,
        image_shape=(224, 224, 3),
    )
    ```
    """

    backbone_cls = None  # set to AltCLIPBackbone after it is defined
