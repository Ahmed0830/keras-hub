"""Convert AltCLIP checkpoints from HuggingFace to KerasHub.

export KAGGLE_USERNAME=XXX
export KAGGLE_KEY=XXX

python tools/checkpoint_conversion/convert_alt_clip_checkpoints.py \
    --preset alt_clip_vit_l14 \
    --upload_uri kaggle://keras/alt_clip/keras/alt_clip_vit_l14
"""

import os

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AltCLIPModel
from transformers import AltCLIPProcessor

import keras_hub
from keras_hub.src.models.alt_clip.alt_clip_backbone import AltCLIPBackbone
from keras_hub.src.models.alt_clip.alt_clip_image_converter import (
    AltCLIPImageConverter,
)
from keras_hub.src.models.alt_clip.alt_clip_preprocessor import (
    AltCLIPPreprocessor,
)
from keras_hub.src.models.alt_clip.alt_clip_text_encoder import (
    AltCLIPTextEncoder,
)
from keras_hub.src.models.alt_clip.alt_clip_tokenizer import AltCLIPTokenizer
from keras_hub.src.models.alt_clip.alt_clip_vision_encoder import (
    AltCLIPVisionEncoder,
)

FLAGS = flags.FLAGS

PRESET_MAP = {
    "alt_clip_vit_l14": "BAAI/AltCLIP",
}

flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
    required=True,
)
flags.DEFINE_string(
    "upload_uri",
    None,
    'Could be "kaggle://keras/alt_clip/keras/alt_clip_vit_l14"',
    required=False,
)


def convert_model(hf_model):
    vision_cfg = hf_model.vision_model.config.to_dict()
    text_cfg = hf_model.text_model.config.to_dict()
    top_cfg = hf_model.config.to_dict()

    image_size = vision_cfg["image_size"]
    vision_encoder = AltCLIPVisionEncoder(
        patch_size=vision_cfg["patch_size"],
        hidden_dim=vision_cfg["hidden_size"],
        num_layers=vision_cfg["num_hidden_layers"],
        num_heads=vision_cfg["num_attention_heads"],
        intermediate_dim=vision_cfg["intermediate_size"],
        intermediate_activation=vision_cfg["hidden_act"],
        image_shape=(image_size, image_size, 3),
    )
    text_encoder = AltCLIPTextEncoder(
        vocabulary_size=text_cfg["vocab_size"],
        num_layers=text_cfg["num_hidden_layers"],
        num_heads=text_cfg["num_attention_heads"],
        hidden_dim=text_cfg["hidden_size"],
        intermediate_dim=text_cfg["intermediate_size"],
        project_dim=text_cfg["project_dim"],
        dropout=text_cfg["hidden_dropout_prob"],
        max_sequence_length=text_cfg["max_position_embeddings"],
        type_vocab_size=text_cfg["type_vocab_size"],
    )
    return AltCLIPBackbone(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        projection_dim=top_cfg["projection_dim"],
        logit_scale_init_value=top_cfg["logit_scale_init_value"],
    )


def convert_weights(keras_hub_model, hf_model):
    state_dict = hf_model.state_dict()
    state_dict.update(dict(hf_model.named_buffers()))

    def port_weights(keras_variable, weight_key, hook_fn=None):
        torch_tensor = state_dict[weight_key].cpu().numpy()
        if hook_fn:
            torch_tensor = hook_fn(torch_tensor, list(keras_variable.shape))
        keras_variable.assign(torch_tensor)

    def port_ln(keras_variable, weight_key):
        port_weights(keras_variable.gamma, f"{weight_key}.weight")
        port_weights(keras_variable.beta, f"{weight_key}.bias")

    def port_dense(keras_variable, weight_key):
        port_weights(
            keras_variable.kernel,
            f"{weight_key}.weight",
            hook_fn=lambda x, _: x.T,
        )
        if keras_variable.bias is not None:
            port_weights(keras_variable.bias, f"{weight_key}.bias")

    def port_mha(keras_variable, weight_key, num_heads, hidden_dim):
        head_dim = hidden_dim // num_heads
        # query
        port_weights(
            keras_variable.query_dense.kernel,
            f"{weight_key}.q_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, head_dim)
            ),
        )
        port_weights(
            keras_variable.query_dense.bias,
            f"{weight_key}.q_proj.bias",
            hook_fn=lambda x, _: np.reshape(x, (num_heads, head_dim)),
        )
        # key
        port_weights(
            keras_variable.key_dense.kernel,
            f"{weight_key}.k_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, head_dim)
            ),
        )
        port_weights(
            keras_variable.key_dense.bias,
            f"{weight_key}.k_proj.bias",
            hook_fn=lambda x, _: np.reshape(x, (num_heads, head_dim)),
        )
        # value
        port_weights(
            keras_variable.value_dense.kernel,
            f"{weight_key}.v_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, head_dim)
            ),
        )
        port_weights(
            keras_variable.value_dense.bias,
            f"{weight_key}.v_proj.bias",
            hook_fn=lambda x, _: np.reshape(x, (num_heads, head_dim)),
        )
        # output projection
        port_weights(
            keras_variable.output_dense.kernel,
            f"{weight_key}.out_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (num_heads, head_dim, hidden_dim)
            ),
        )
        port_weights(
            keras_variable.output_dense.bias, f"{weight_key}.out_proj.bias"
        )

    # ── Vision encoder ──────────────────────────────────────────────────────
    vision_encoder = keras_hub_model.vision_encoder

    port_weights(
        vision_encoder.embedding.patch_embedding.kernel,
        "vision_model.embeddings.patch_embedding.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )
    port_weights(
        vision_encoder.embedding.position_embedding.embeddings,
        "vision_model.embeddings.position_embedding.weight",
    )
    port_weights(
        vision_encoder.embedding.class_embedding,
        "vision_model.embeddings.class_embedding",
    )
    port_weights(
        vision_encoder.embedding.position_ids,
        "vision_model.embeddings.position_ids",
    )
    port_ln(vision_encoder.pre_layer_norm, "vision_model.pre_layrnorm")

    for i, enc_layer in enumerate(vision_encoder.encoder_layers):
        prefix = f"vision_model.encoder.layers.{i}"
        num_heads = enc_layer.num_heads
        hidden_dim = enc_layer.hidden_dim
        port_ln(enc_layer.layer_norm_1, f"{prefix}.layer_norm1")
        port_mha(
            enc_layer.attention, f"{prefix}.self_attn", num_heads, hidden_dim
        )
        port_ln(enc_layer.layer_norm_2, f"{prefix}.layer_norm2")
        port_dense(enc_layer.dense_1, f"{prefix}.mlp.fc1")
        port_dense(enc_layer.dense_2, f"{prefix}.mlp.fc2")

    port_ln(vision_encoder.layer_norm, "vision_model.post_layernorm")

    # ── Text encoder ────────────────────────────────────────────────────────
    # AltCLIP's text model uses AltRobertaModel (XLM-RoBERTa-style) inside
    # AltCLIPTextModel. HF keys:
    #   text_model.roberta.embeddings.*
    #   text_model.roberta.encoder.layer.{i}.*  (note: "layer", not "layers")
    #   text_model.pre_LN.*
    #   text_model.transformation.*
    text_encoder = keras_hub_model.text_encoder
    text_prefix = "text_model.roberta"

    port_weights(
        text_encoder.embeddings.token_embedding._embeddings,
        f"{text_prefix}.embeddings.word_embeddings.weight",
    )
    port_weights(
        text_encoder.embeddings.position_embedding.position_embeddings,
        f"{text_prefix}.embeddings.position_embeddings.weight",
    )
    port_ln(
        text_encoder.embeddings_layer_norm,
        f"{text_prefix}.embeddings.LayerNorm",
    )

    for i, transformer_layer in enumerate(text_encoder.transformer_layers):
        prefix = f"{text_prefix}.encoder.layers.{i}"
        num_heads = transformer_layer._self_attention_layer.num_heads
        hidden_dim = (
            transformer_layer._self_attention_layer.num_heads
            * transformer_layer._self_attention_layer.key_dim
        )

        # Self-attention weights in HF AltRoberta use separate query/key/value
        # Linear modules (not combined q_k_v).
        port_weights(
            transformer_layer._self_attention_layer.query_dense.kernel,
            f"{prefix}.attention.self.query.weight",
            hook_fn=lambda x, s: np.reshape(x.T, s),
        )
        port_weights(
            transformer_layer._self_attention_layer.query_dense.bias,
            f"{prefix}.attention.self.query.bias",
            hook_fn=lambda x, s: np.reshape(x, s),
        )
        port_weights(
            transformer_layer._self_attention_layer.key_dense.kernel,
            f"{prefix}.attention.self.key.weight",
            hook_fn=lambda x, s: np.reshape(x.T, s),
        )
        port_weights(
            transformer_layer._self_attention_layer.key_dense.bias,
            f"{prefix}.attention.self.key.bias",
            hook_fn=lambda x, s: np.reshape(x, s),
        )
        port_weights(
            transformer_layer._self_attention_layer.value_dense.kernel,
            f"{prefix}.attention.self.value.weight",
            hook_fn=lambda x, s: np.reshape(x.T, s),
        )
        port_weights(
            transformer_layer._self_attention_layer.value_dense.bias,
            f"{prefix}.attention.self.value.bias",
            hook_fn=lambda x, s: np.reshape(x, s),
        )
        port_weights(
            transformer_layer._self_attention_layer.output_dense.kernel,
            f"{prefix}.attention.output.dense.weight",
            hook_fn=lambda x, s: np.reshape(x.T, s),
        )
        port_weights(
            transformer_layer._self_attention_layer.output_dense.bias,
            f"{prefix}.attention.output.dense.bias",
        )
        port_ln(
            transformer_layer._self_attention_layer_norm,
            f"{prefix}.attention.output.LayerNorm",
        )
        # FFN
        port_dense(
            transformer_layer._feedforward_intermediate_dense,
            f"{prefix}.intermediate.dense",
        )
        port_dense(
            transformer_layer._feedforward_output_dense,
            f"{prefix}.output.dense",
        )
        port_ln(
            transformer_layer._feedforward_layer_norm,
            f"{prefix}.output.LayerNorm",
        )

    port_ln(text_encoder.pre_projection_layer_norm, "text_model.pre_LN")
    port_dense(text_encoder.projection, "text_model.transformation")

    # ── Top-level projection layers & logit scale ───────────────────────────
    port_weights(
        keras_hub_model.visual_projection.kernel,
        "visual_projection.weight",
        hook_fn=lambda x, _: x.T,
    )
    port_weights(
        keras_hub_model.text_projection.kernel,
        "text_projection.weight",
        hook_fn=lambda x, _: x.T,
    )
    port_weights(keras_hub_model.clip_head.logit_scale, "logit_scale")


def convert_image_converter(hf_processor):
    cfg = hf_processor.image_processor.to_dict()
    image_size = (
        cfg["crop_size"]["height"],
        cfg["crop_size"]["width"],
    )
    std = cfg["image_std"]
    mean = cfg["image_mean"]
    return AltCLIPImageConverter(
        image_size=image_size,
        scale=[1.0 / 255.0 / s for s in std],
        offset=[-m / s for m, s in zip(mean, std)],
        interpolation="bicubic",
    )


def convert_tokenizer(hf_preset):
    proto_path = hf_hub_download(
        hf_preset, "sentencepiece.bpe.model", token=True
    )
    return AltCLIPTokenizer(proto=proto_path)


def validate_output(
    keras_model,
    keras_image_converter,
    keras_tokenizer,
    hf_model,
    hf_processor,
):
    file = keras.utils.get_file(
        origin="http://images.cocodataset.org/val2017/000000039769.jpg"
    )
    image = Image.open(file)
    text = ["a photo of a cat", "a photo of a dog"]

    # HF preprocessing.
    hf_inputs = hf_processor(
        text=text,
        images=[image, image],
        return_tensors="pt",
    )

    # KerasHub preprocessing.
    images_np = np.expand_dims(np.array(image).astype("float32"), axis=0)
    images_np = np.concatenate([images_np, images_np], axis=0)
    keras_images = keras_image_converter(images_np)
    keras_preprocessor = AltCLIPPreprocessor(keras_tokenizer)
    keras_token_ids = keras_preprocessor({"prompts": text, "images": None})[
        "token_ids"
    ]

    # Replace HF pixel values with keras-processed values for a fair
    # model comparison that isolates weight correctness.
    hf_inputs["pixel_values"] = torch.from_numpy(
        keras.ops.convert_to_numpy(
            keras.ops.transpose(keras_images, (0, 3, 1, 2))
        )
    )
    hf_inputs["input_ids"] = torch.from_numpy(
        keras.ops.convert_to_numpy(keras_token_ids)
    )

    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs)
    hf_logits = hf_outputs.logits_per_image.cpu().numpy()

    keras_outputs = keras_model(
        {
            "images": keras_images,
            "token_ids": keras_token_ids,
        }
    )
    keras_logits = keras.ops.convert_to_numpy(keras_outputs["vision_logits"])

    print("HF vision logits:", hf_logits)
    print("Keras vision logits:", keras_logits)
    np.testing.assert_allclose(hf_logits, keras_logits, atol=1e-3)
    print("✓ Output validation passed.")


def main(_):
    hf_preset = PRESET_MAP[FLAGS.preset]
    preset_name = FLAGS.preset

    print(f"\n-> Loading HuggingFace model '{hf_preset}'...")
    hf_model = AltCLIPModel.from_pretrained(
        hf_preset, token=True, attn_implementation="eager"
    )
    hf_model.eval()
    hf_processor = AltCLIPProcessor.from_pretrained(hf_preset, token=True)

    print("-> Converting model architecture...")
    keras_model = convert_model(hf_model)

    print("-> Porting weights...")
    convert_weights(keras_model, hf_model)

    print("-> Converting image converter...")
    keras_image_converter = convert_image_converter(hf_processor)

    print("-> Converting tokenizer...")
    keras_tokenizer = convert_tokenizer(hf_preset)

    # print("-> Validating outputs...")
    # validate_output(
    #     keras_model,
    #     keras_image_converter,
    #     keras_tokenizer,
    #     hf_model,
    #     hf_processor,
    # )

    keras_model.save_to_preset(f"./{preset_name}")
    keras_image_converter.save_to_preset(f"./{preset_name}")
    keras_tokenizer.save_to_preset(f"./{preset_name}")
    print(f"Preset saved to ./{preset_name}.")

    if FLAGS.upload_uri:
        keras_hub.upload_preset(uri=FLAGS.upload_uri, preset=f"./{preset_name}")
        print(f"Preset uploaded to {FLAGS.upload_uri}.")

    print("Done.")


if __name__ == "__main__":
    app.run(main)
