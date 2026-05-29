import os

import numpy as np
import pytest

from keras_hub.src.models.alt_clip.alt_clip_image_converter import (
    AltCLIPImageConverter,
)
from keras_hub.src.models.alt_clip.alt_clip_preprocessor import (
    AltCLIPPreprocessor,
)
from keras_hub.src.models.alt_clip.alt_clip_tokenizer import AltCLIPTokenizer
from keras_hub.src.tests.test_case import TestCase


class AltCLIPPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = AltCLIPTokenizer(
            proto=os.path.join(
                self.get_test_data_dir(), "xlm_roberta_test_vocab.spm"
            )
        )
        self.image_converter = AltCLIPImageConverter(
            (224, 224),
            [2.0 / 255.0] * 3,
            [-1.0] * 3,
            interpolation="bicubic",
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": self.image_converter,
            "sequence_length": 8,
        }
        self.input_data = {
            "prompts": ["the quick brown fox"],
            "images": [np.zeros([512, 512, 3])],
        }

    def test_preprocessor_basics(self):
        self.run_preprocessing_layer_test(
            cls=AltCLIPPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_without_images(self):
        input_data = {
            "prompts": ["the quick brown fox"] * 2,
            "images": None,
        }
        preprocessor = AltCLIPPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self.image_converter,
            sequence_length=8,
            add_start_token=False,
            add_end_token=False,
        )
        x = preprocessor(input_data)
        self.assertIsNone(x["images"])

    def test_no_start_end_token(self):
        input_data = {
            "prompts": ["the quick brown fox"] * 2,
            "images": [np.zeros([512, 512, 3])],
        }
        preprocessor = AltCLIPPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self.image_converter,
            sequence_length=8,
            add_start_token=False,
            add_end_token=False,
        )
        x = preprocessor(input_data)
        # Check that outputs have the correct shape.
        self.assertEqual(x["token_ids"].shape[-1], 8)

    def test_sequence_length_override(self):
        input_data = {
            "prompts": "the quick brown fox",
            "images": [np.zeros([512, 512, 3])],
        }
        preprocessor = AltCLIPPreprocessor(**self.init_kwargs)
        x = preprocessor(input_data, sequence_length=5)
        self.assertEqual(x["token_ids"].shape[-1], 5)

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        self.skipTest("TODO")
        for preset in AltCLIPPreprocessor.presets:
            self.run_preset_test(
                cls=AltCLIPPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
