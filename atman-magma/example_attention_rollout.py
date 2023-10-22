from atman_magma.magma  import Magma
from atman.attention_rollout import AttentionRollout

print('loading model...')
model = Magma.from_checkpoint(
    checkpoint_path = "./mp_rank_00_model_states.pt",
    device = 'cuda:0'
)
attention_rollout = AttentionRolloutMagma(model = model)


prompt = ['hello world my name is Ben and i love burgers']
embeddings = attention_rollout.model.preprocess_inputs(prompt) ## prompt is a list

output = model.generate(
    embeddings = embeddings,
    max_steps = 1,
    temperature = 0.00001,
    top_k = 1,
    top_p = 0.0,
)
print('Completion:',output[0])

y = attention_rollout.run(embeddings)

print(y.shape) ##shape: (num_layers, sequence)
