from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.xlm_roberta.xlm_roberta_tokenizer import (
    XLMRobertaTokenizer,
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.AltCLIPTokenizer",
        "keras_hub.models.AltCLIPTokenizer",
    ]
)
class AltCLIPTokenizer(XLMRobertaTokenizer):
    """AltCLIP tokenizer (XLM-RoBERTa SentencePiece tokenizer).

    This tokenizer class tokenizes raw strings into integer sequences and is
    based on `keras_hub.models.XLMRobertaTokenizer`. It uses a SentencePiece
    model trained for the XLM-RoBERTa multilingual vocabulary and applies the
    same fairseq ID shift as the parent tokenizer.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        proto: Either a `string` path to a SentencePiece proto file or a
            `bytes` object with a serialized SentencePiece proto. See the
            [SentencePiece repository](https://github.com/google/sentencepiece)
            for more details on the format.

    Examples:
    ```python
    tokenizer = keras_hub.models.AltCLIPTokenizer.from_preset(
        "alt_clip_vit_l14",
    )

    # Unbatched inputs.
    tokenizer("the quick brown fox")

    # Batched inputs.
    tokenizer(["the quick brown fox", "速度与激情"])

    # Detokenization.
    tokenizer.detokenize(tokenizer("the quick brown fox"))
    ```
    """

    backbone_cls = None  # set to AltCLIPBackbone after it is defined
