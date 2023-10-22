import torch
from transformers import  AutoConfig
from .gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM
from transformers.modeling_utils import no_init_weights

LANGUAGE_MODELS = [
    "gptj",
]


def gptj_config():
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
    config.gradient_checkpointing = True
    return config


def get_gptj(
    gradient_checkpointing: bool = True,
    from_pretrained=False,
    config = None
) -> torch.nn.Module:
    """
    Loads GPTJ language model from HF
    """
    if config is None:
        config = gptj_config()
    config.gradient_checkpointing = gradient_checkpointing
    if gradient_checkpointing:
        config.use_cache = False
    config.model_device = "cpu"
    if from_pretrained:
        raise NotImplemented("GPTJ pretrained not implemented")
    else:
        with no_init_weights():
            model = GPTNeoForCausalLM(config=config)
    return model
