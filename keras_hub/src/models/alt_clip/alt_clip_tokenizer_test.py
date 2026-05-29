import os

import pytest

from keras_hub.src.models.alt_clip.alt_clip_tokenizer import AltCLIPTokenizer
from keras_hub.src.tests.test_case import TestCase


class AltCLIPTokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            # Reuse the XLM-RoBERTa test SentencePiece proto.
            "proto": os.path.join(
                self.get_test_data_dir(), "xlm_roberta_test_vocab.spm"
            )
        }
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=AltCLIPTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[6, 11, 7, 9], [6, 8, 10, 12]],
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=AltCLIPTokenizer,
            preset="alt_clip_vit_l14",
            input_data=["The quick brown fox."],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in AltCLIPTokenizer.presets:
            self.run_preset_test(
                cls=AltCLIPTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
