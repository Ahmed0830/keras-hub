import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.alt_clip.alt_clip_backbone import AltCLIPBackbone
from keras_hub.src.models.alt_clip.alt_clip_image_converter import (
    AltCLIPImageConverter,
)
from keras_hub.src.models.alt_clip.alt_clip_tokenizer import AltCLIPTokenizer
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.AltCLIPPreprocessor")
class AltCLIPPreprocessor(CausalLMPreprocessor):
    """AltCLIP preprocessor.

    This preprocessing layer prepares inputs for `AltCLIPBackbone`. By
    default, it takes in batches of strings and images and returns token ids
    and resized images.

    Args:
        tokenizer: An `AltCLIPTokenizer` instance.
        image_converter: An `AltCLIPImageConverter` instance.
        sequence_length: int. The length of the packed token sequences.
            Defaults to `77`.
        add_start_token: bool. If `True`, the preprocessor will prepend the
            tokenizer start token to each input sequence. Defaults to `True`.
        add_end_token: bool. If `True`, the preprocessor will append the
            tokenizer end token to each input sequence. Defaults to `True`.

    Call arguments:
        x: A dict with `"prompts"` and `"images"` keys, where `"prompts"` is
            a `tf.Tensor` or list of Python strings and `"images"` are the
            image tensors.
        y: Label data. Should always be `None`.
        sample_weight: Label weights.
        sequence_length: Pass to override the configured `sequence_length` of
            the layer.

    Examples:
    ```python
    # Load the preprocessor from a preset.
    preprocessor = keras_hub.models.AltCLIPPreprocessor.from_preset(
        "alt_clip_vit_l14"
    )

    # Tokenize the sentence and preprocess the image.
    preprocessor(
        {
            "prompts": "The quick brown fox jumped.",
            "images": np.ones(shape=(123, 123, 3)),
        }
    )

    # Tokenize a batch of sentences and preprocess a batch of images.
    preprocessor(
        {
            "prompts": [
                "The quick brown fox jumped.",
                "一只棕色的狐狸跳过了。"
            ],
            "images": np.ones(shape=(2, 123, 123, 3)),
        }
    )
    ```
    """

    backbone_cls = AltCLIPBackbone
    tokenizer_cls = AltCLIPTokenizer
    image_converter_cls = AltCLIPImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        sequence_length=77,
        add_start_token=True,
        add_end_token=True,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            add_start_token=add_start_token,
            add_end_token=add_end_token,
            **kwargs,
        )
        self.image_converter = image_converter

    def build(self, input_shape):
        # Defer packer creation to `build()` so that tokenizer assets have
        # loaded when restoring a saved model.
        self.packer = StartEndPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.sequence_length,
            return_padding_mask=True,
        )
        self.built = True

    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        sequence_length = sequence_length or self.sequence_length
        images, prompts = x["images"], x["prompts"]
        prompts = self.tokenizer(prompts)
        if images is not None and self.image_converter:
            images = self.image_converter(images)
        token_ids, padding_mask = self.packer(
            prompts,
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        x = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "images": images,
        }
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
