import matplotlib.pyplot as plt

from magma.image_input import ImageInput
from atman_magma.chefer.method import CheferMethod
from atman_magma.chefer.chefer_magma.magma import CheferMagma

device = 'cuda:0'
model = CheferMagma.from_checkpoint(
    checkpoint_path = "./mp_rank_00_model_states.pt",
    device = device
)

cm = CheferMethod(
    magma=model,
    device=device
)

prompt =[
    ## supports urls and path/to/image
    ImageInput('https://www.art-prints-on-demand.com/kunst/thomas_cole/woods_hi.jpg'),
    'This is a picture of a'
]
embeddings = model.preprocess_inputs(prompt)

relevance_maps = cm.run(
    embeddings = embeddings, 
    target = ' cabin in the woods'
)

fig = plt.figure()
plt.imshow(relevance_maps[0]['relevance_map'].reshape(12,12))
plt.show()
fig.savefig('chefer.jpg')
