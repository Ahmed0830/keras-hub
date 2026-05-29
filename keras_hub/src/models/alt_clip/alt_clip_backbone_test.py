import pytest
from keras import ops

from keras_hub.src.models.alt_clip.alt_clip_backbone import AltCLIPBackbone
from keras_hub.src.models.alt_clip.alt_clip_text_encoder import (
    AltCLIPTextEncoder,
)
from keras_hub.src.models.alt_clip.alt_clip_vision_encoder import (
    AltCLIPVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class AltCLIPBackboneTest(TestCase):
    def setUp(self):
        vision_encoder = AltCLIPVisionEncoder(
            patch_size=2,
            hidden_dim=8,
            num_layers=2,
            num_heads=2,
            intermediate_dim=16,
            image_shape=(4, 4, 3),
            name="vision_encoder",
        )
        text_encoder = AltCLIPTextEncoder(
            vocabulary_size=16,
            num_layers=2,
            num_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            project_dim=8,
            dropout=0.0,
            max_sequence_length=8,
            name="text_encoder",
        )
        self.init_kwargs = {
            "vision_encoder": vision_encoder,
            "text_encoder": text_encoder,
            "projection_dim": 8,
        }
        self.input_data = {
            "images": ops.ones((2, 4, 4, 3)),
            "token_ids": ops.ones((2, 8), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=AltCLIPBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "vision_logits": (2, 2),
                "text_logits": (2, 2),
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=AltCLIPBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in AltCLIPBackbone.presets:
            self.run_preset_test(
                cls=AltCLIPBackbone,
                preset=preset,
                input_data=self.input_data,
            )
