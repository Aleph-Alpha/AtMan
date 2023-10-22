'''
Forked from: https://github.com/Aleph-Alpha/magma/blob/master/magma/sampling.py
'''
import torch
import torch.nn.functional as F
from torchtyping import TensorType
from typing import Union, List

from magma.sampling import (
    top_p_filter,
    top_k_filter,
    remove_tokens_after_eos
)


@torch.no_grad()
def generate_with_atman(
    model: "Magma",
    embeddings: TensorType["b", "s", "d"],
    max_steps: int = 100,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 0.9,
    eos_token: int = None,
    decode: bool = True,
    suppression_token_indices: List[List[int]] = None,
    suppression_factors: List[List[float]] = None
) -> Union[List[str], TensorType["b", "s"]]:
    """
    Generates captions for a batch of embeddings.
    :param model: The model to use for generation.
    :param embeddings: The embeddings to generate captions for.
    :param max_steps: The maximum number of steps to generate captions for.
    :param temperature: The temperature to use for sampling.
    :param top_k: value for top k sampling. If 0, no sampling will be used.
    :param top_p: value for top p sampling. If 0, no sampling will be used.
    :param eos_token: The token to use for end of sequence.
    :param decode: Whether to decode the output into text, or return the raw tokens.
    """

    # init values
    eos_token = eos_token or model.eos_token
    was_training = model.training
    model.eval()
    b, s, _ = embeddings.shape
    past_key_values = None

    # init output with image tokens
    out = torch.zeros((b, s), dtype=torch.long).to(model.device) + model.image_token

    if suppression_token_indices is not None and suppression_factors is not None:
        model.lm.suppression_token_indices = suppression_token_indices
        model.lm.suppression_factors = suppression_factors

    # do sampling
    for i in range(max_steps):

        if i == 0:
            # initial input
            outputs = model.lm(
                inputs_embeds=embeddings,
                use_cache=False,
                past_key_values=None,
            )
    
        else:
            next_token_embedding = model.word_embedding(next_token)
            embeddings = torch.cat([embeddings, next_token_embedding], dim = 1)
            outputs = model.lm(
                inputs_embeds=embeddings,
                use_cache=False,
                past_key_values=None,
            )

        
        logits = outputs.logits[:, -1, :].float()
        past_key_values = outputs.past_key_values

        # filter / temperature sample
        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1)
        else:
            if top_k > 0:
                logits = top_k_filter(logits, k=top_k)
            if top_p > 0:
                logits = top_p_filter(logits, threshold=top_p)

            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        out = torch.cat((out, next_token), dim=-1)

        if eos_token is not None and (next_token == eos_token).all():
            break
            
    # ## cleanup
    # model.lm.suppression_token_indices = None
    # model.lm.suppression_factors = None


    if decode:
        captions = []
        for b in out:
            b = remove_tokens_after_eos(b, eos_token, model.image_token)
            caption = model.tokenizer.decode(b)
            captions.append(caption)
        out = captions

    model.train(was_training)
    return out