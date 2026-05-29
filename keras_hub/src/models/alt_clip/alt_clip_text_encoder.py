from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.token_and_position_embedding import (
    TokenAndPositionEmbedding,
)
from keras_hub.src.layers.modeling.transformer_encoder import TransformerEncoder
from keras_hub.src.models.backbone import Backbone


@keras_hub_export("keras_hub.models.AltCLIPTextEncoder")
class AltCLIPTextEncoder(Backbone):
    """AltCLIP text encoder built on XLM-RoBERTa transformer layers.

    This backbone implements the text encoding component of AltCLIP. It is a
    post-norm RoBERTa-style transformer (GELU activation, LayerNorm after
    residual) with an additional pre-projection LayerNorm and a final Dense
    projection applied to the CLS token.

    The default constructor gives a fully customizable, randomly initialized
    encoder. To load preset architectures and weights, use the `from_preset()`
    constructor.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The size of the transformer encoding layer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        project_dim: int. The output dimension of the final projection Dense
            layer applied to the CLS token. Defaults to `768`.
        dropout: float. Dropout probability for the Transformer encoder.
            Defaults to `0.1`.
        max_sequence_length: int. The maximum sequence length this encoder can
            consume. Defaults to `514`.
        type_vocab_size: int. The number of token types (unused in practice for
            AltCLIP; kept for config compatibility). Defaults to `1`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the models computations and weights. Note that some
            computations, such as softmax and layer normalization will always
            be done at float32 precision regardless of dtype.

    Example:
    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Randomly initialized AltCLIPTextEncoder with custom config.
    encoder = keras_hub.models.AltCLIPTextEncoder(
        vocabulary_size=250002,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        intermediate_dim=4096,
        project_dim=768,
    )
    encoder(input_data)
    ```
    """

    backbone_cls = None  # set to AltCLIPBackbone after it is defined

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        project_dim=768,
        dropout=0.1,
        max_sequence_length=514,
        type_vocab_size=1,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.embeddings = TokenAndPositionEmbedding(
            vocabulary_size=vocabulary_size,
            sequence_length=max_sequence_length,
            embedding_dim=hidden_dim,
            dtype=dtype,
            name="embeddings",
        )
        self.embeddings_layer_norm = layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            dtype=dtype,
            name="embeddings_layer_norm",
        )
        self.embeddings_dropout = layers.Dropout(
            dropout,
            dtype=dtype,
            name="embeddings_dropout",
        )
        self.transformer_layers = [
            TransformerEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                activation="gelu",
                dropout=dropout,
                layer_norm_epsilon=1e-5,
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.pre_projection_layer_norm = layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            dtype=dtype,
            name="pre_projection_layer_norm",
        )
        self.projection = layers.Dense(
            project_dim,
            dtype=dtype,
            name="projection",
        )

        # === Functional Model ===
        token_id_input = layers.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = layers.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        x = self.embeddings(token_id_input)
        x = self.embeddings_layer_norm(x)
        x = self.embeddings_dropout(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, padding_mask=padding_mask_input)
        x = self.pre_projection_layer_norm(x)
        # CLS token pooling: take the first token (index 0).
        cls_output = x[:, 0, :]
        output = self.projection(cls_output)

        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.project_dim = project_dim
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.type_vocab_size = type_vocab_size

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "project_dim": self.project_dim,
                "dropout": self.dropout,
                "max_sequence_length": self.max_sequence_length,
                "type_vocab_size": self.type_vocab_size,
            }
        )
        return config
