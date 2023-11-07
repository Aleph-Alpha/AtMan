from atman_magma.magma  import Magma
from atman_magma.attention_rollout import AttentionRolloutMagma
import matplotlib.pyplot as plt

from magma.image_input import ImageInput
import  PIL.Image as PilImage


print('loading model...')
model = Magma.from_checkpoint(
    checkpoint_path = "./mp_rank_00_model_states.pt",
    device = 'cuda:0'
)
ar = AttentionRolloutMagma(model = model)



prompt =[
    ## supports urls and path/to/image
    #ImageInput('https://www.art-prints-on-demand.com/kunst/thomas_cole/woods_hi.jpg'),
    ImageInput('',pil=PilImage.open('openimages-panda.jpg')),
    'This is a picture of a'
]

relevance_maps = ar.run_on_image(
    prompt=prompt,
    target = 'Panda', # note rollout per se does not have a target
)

fig = plt.figure()
plt.imshow(relevance_maps.reshape(12,12))
fig.savefig('panda-explained-rollout.jpg')
print('panda-explained-rollout.jpg')
