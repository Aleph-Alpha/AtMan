""" 
Taken and modified from:
https://github.com/finetuneanon/transformers
"""

from transformers import AutoConfig

def get_gptj_config():
    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B")
    config.attention_layers = ["global"] * 28
    config.attention_types = [["global"], 28]
    config.num_layers = 28
    config.num_heads = 16
    config.hidden_size = 256 * config.num_heads
    config.vocab_size = 50400
    config.rotary = True
    config.rotary_dim = 64
    config.jax = True
    config.gradient_checkpointing = False

    ## magma config stuff
    config.encoder_name: str = "clip_resnet_large"
    config.tokenizer_name: str = "gpt2"
    config.lm_name: str = "EleutherAI/gpt-j-6B"
    config.image_seq_len: int = 2
    config.pretrained_img_encoder: bool = False
    config.seq_len: int = None

    # Layer Freezing settings:
    # ------------------------------------------------------------
    config.freeze_lm: bool = True
    config.freeze_img_encoder: bool = True

    config.image_embed_dropout_prob: float = 0.0
    config.use_image_embed_layernorm: bool = False

    # Adapter settings:
    # ------------------------------------------------------------
    config.adapter_config: dict = {"mlp": {"adapter_type": "normal", "downsample_factor": 4}}

    # Classification Finetuning settings:
    # ------------------------------------------------------------
    config.class_dict: dict = None  # {num_classes: .., ckpt_path: .., classifier_type:, .., interface_type: .., interface_position: .., freeze_model: ..}
    config.image_size: int = 384
    return config