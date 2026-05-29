from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.clip.clip_layers import CLIPHead
from keras_hub.src.models.clip.clip_layers import CLIPVisionPooler


@keras_hub_export("keras_hub.models.AltCLIPBackbone")
class AltCLIPBackbone(Backbone):
    """AltCLIP core network with hyperparameters.

    This backbone implements the base architecture for AltCLIP, a bilingual
    (English + Chinese) vision-language model. It combines an
    `AltCLIPVisionEncoder` (identical to CLIP ViT) with an
    `AltCLIPTextEncoder` (XLM-RoBERTa-style post-norm transformer) and
    produces paired vision and text logits via cosine similarity.

    The default constructor gives a fully customizable, randomly initialized
    AltCLIP model. To load preset architectures and weights, use the
    `from_preset()` constructor.

    Args:
        vision_encoder: An `AltCLIPVisionEncoder` instance for encoding images.
        text_encoder: An `AltCLIPTextEncoder` instance for encoding token
            sequences.
        projection_dim: int. The output dimensionality of both the vision and
            text projection layers. Defaults to `768`.
        logit_scale_init_value: float. Initial value for the log of the logit
            scale used in the contrastive head. Defaults to `2.6592`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the models computations and weights. Note that some
            computations, such as softmax and layer normalization will always
            be done at float32 precision regardless of dtype.

    Example:
    ```python
    input_data = {
        "images": np.ones(shape=(1, 224, 224, 3), dtype="float32"),
        "token_ids": np.ones(shape=(1, 77), dtype="int32"),
    }

    # Pretrained AltCLIP model.
    model = keras_hub.models.AltCLIPBackbone.from_preset("alt_clip_vit_l14")
    model(input_data)

    # Randomly initialized AltCLIP model with custom config.
    vision_encoder = keras_hub.models.AltCLIPVisionEncoder(
        patch_size=14,
        hidden_dim=1024,
        num_layers=24,
        num_heads=16,
        intermediate_dim=4096,
        image_shape=(224, 224, 3),
    )
    text_encoder = keras_hub.models.AltCLIPTextEncoder(
        vocabulary_size=250002,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        intermediate_dim=4096,
        project_dim=768,
    )
    model = keras_hub.models.AltCLIPBackbone(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        projection_dim=768,
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vision_encoder,
        text_encoder,
        projection_dim=768,
        logit_scale_init_value=2.6592,
        dtype=None,
        name=None,
        **kwargs,
    ):
        # === Layers ===
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.vision_pooler = CLIPVisionPooler(dtype=dtype, name="vision_pooler")
        self.visual_projection = layers.Dense(
            projection_dim,
            use_bias=False,
            dtype=dtype,
            name="visual_projection",
        )
        self.text_projection = layers.Dense(
            projection_dim,
            use_bias=False,
            dtype=dtype,
            name="text_projection",
        )
        self.clip_head = CLIPHead(dtype=dtype, name="clip_head")

        # === Functional Model ===
        image_input = layers.Input(
            shape=self.vision_encoder.image_shape, name="images"
        )
        token_id_input = layers.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        vision_embeddings = self.get_vision_embeddings(image_input)
        text_embeddings = self.get_text_embeddings(token_id_input)
        vision_logits, text_logits = self.clip_head(
            vision_embeddings, text_embeddings
        )

        super().__init__(
            inputs={
                "images": image_input,
                "token_ids": token_id_input,
            },
            outputs={
                "vision_logits": vision_logits,
                "text_logits": text_logits,
            },
            dtype=dtype,
            name=name,
            **kwargs,
        )

        # === Config ===
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value

    def get_vision_embeddings(self, images):
        """Get the embeddings from the vision encoder.

        Args:
            images: The input tensor for the vision encoder.

        Returns:
            The output embeddings obtained by applying the visual projection
            layer to the pooled output of the vision encoder.
        """
        vision_outputs = self.vision_encoder({"images": images})
        vision_outputs = self.vision_pooler(vision_outputs)
        return self.visual_projection(vision_outputs)

    def get_text_embeddings(self, token_ids):
        """Get the embeddings from the text encoder.

        The padding mask is computed from `token_ids` by marking positions
        where the pad token id (1) is not present.

        Args:
            token_ids: The input int tensor for the text encoder.

        Returns:
            The output embeddings obtained by applying the text projection
            layer to the projected CLS output of the text encoder.
        """
        padding_mask = ops.cast(ops.not_equal(token_ids, 1), dtype="int32")
        text_outputs = self.text_encoder(
            {"token_ids": token_ids, "padding_mask": padding_mask}
        )
        return self.text_projection(text_outputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_encoder": layers.serialize(self.vision_encoder),
                "text_encoder": layers.serialize(self.text_encoder),
                "projection_dim": self.projection_dim,
                "logit_scale_init_value": self.logit_scale_init_value,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()

        # Propagate `dtype` to submodels if needed.
        if "dtype" in config and config["dtype"] is not None:
            dtype_config = config["dtype"]
            if "dtype" not in config["vision_encoder"]["config"]:
                config["vision_encoder"]["config"]["dtype"] = dtype_config
            if "dtype" not in config["text_encoder"]["config"]:
                config["text_encoder"]["config"]["dtype"] = dtype_config

        # Deserialize submodels.
        config["vision_encoder"] = layers.deserialize(
            config["vision_encoder"], custom_objects=custom_objects
        )
        config["text_encoder"] = layers.deserialize(
            config["text_encoder"], custom_objects=custom_objects
        )
        return cls(**config)


# Resolve forward references: set backbone_cls on encoders and tokenizer
# after AltCLIPBackbone is defined.
from keras_hub.src.models.alt_clip.alt_clip_text_encoder import (  # noqa: E402
    AltCLIPTextEncoder,
)
from keras_hub.src.models.alt_clip.alt_clip_tokenizer import (  # noqa: E402
    AltCLIPTokenizer,
)
from keras_hub.src.models.alt_clip.alt_clip_vision_encoder import (  # noqa: E402
    AltCLIPVisionEncoder,
)

AltCLIPVisionEncoder.backbone_cls = AltCLIPBackbone
AltCLIPTextEncoder.backbone_cls = AltCLIPBackbone
AltCLIPTokenizer.backbone_cls = AltCLIPBackbone
