from atman_magma.magma  import Magma
from atman_magma.explainer import Explainer
from atman_magma.utils import split_str_into_tokens
from atman_magma.logit_parsing import get_delta_cross_entropies


print('loading model...')
device = 'cuda:0'
model = Magma.from_checkpoint(
    checkpoint_path = "./mp_rank_00_model_states.pt",
    device = device
)


'''
Image example
'''
from magma.image_input import ImageInput
import  PIL.Image as PilImage

ex = Explainer(
    model = model,
    device = device,
    tokenizer = model.tokenizer,
#     conceptual_suppression_threshold = None
    conceptual_suppression_threshold = 0.75
)


prompt =[
    ## supports urls and path/to/image
    ImageInput('',pil=PilImage.open('openimages-panda.jpg')),
    'This is a picture of a'
]

## returns a tensor of shape: (1, 149, 4096)
embeddings = model.preprocess_inputs(prompt.copy())

## returns a list of length embeddings.shape[0] (batch size)
# output = model.generate(
#     embeddings = embeddings,
#     max_steps = 5,
#     temperature = 0.001,
#     top_k = 1,
#     top_p = 0.0,
# )
# completion = output[0]

logit_outputs = ex.collect_logits_by_manipulating_attention(
    prompt = prompt.copy(),
    target = 'Panda',
    max_batch_size=1,
    # prompt_explain_indices=[i for i in range(10)]
)

results = get_delta_cross_entropies(
    output = logit_outputs
)

results.save('output.json')