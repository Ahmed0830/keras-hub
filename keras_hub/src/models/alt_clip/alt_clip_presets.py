"""AltCLIP model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "alt_clip_vit_l14": {
        "metadata": {
            "description": (
                "AltCLIP model with a ViT-L/14 vision encoder and "
                "XLM-RoBERTa-Large text encoder. Trained by BAAI on bilingual "
                "(English + Chinese) image-text pairs."
            ),
            "params": 0,  # fill in after conversion
            "path": "alt_clip",
        },
        "kaggle_handle": "kaggle://adamda0830/alt-clip/keras/alt_clip_vit_l14",
    }
}
