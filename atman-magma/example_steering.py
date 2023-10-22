from atman_magma.magma  import Magma

print('loading model...')
model = Magma.from_checkpoint(
    checkpoint_path = "./mp_rank_00_model_states.pt",
    device = 'cuda:0'
)

prompt = 'The worst thing about Trump is that he'
embeddings = model.preprocess_inputs([prompt])

## returns a list of length embeddings.shape[0] (batch size)
output = model.generate(
    embeddings = embeddings,
    max_steps = 20,
    temperature = 0.00001,
    top_k = 1,
    top_p = 0.0,
)
print('Prompt:', prompt)
print('Completion:',output[0])
print('Now suppressing token: <worst>')

output = model.generate_steered(
    embeddings = embeddings,
    max_steps = 20,
    temperature = 0.00001,
    top_k = 1,
    top_p = 0.0,
    suppression_factors=[
        [0.0]
    ],
    suppression_token_indices=[
        [1]
    ],
)

print('Completion:',output[0])
