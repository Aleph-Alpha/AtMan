from PIL import Image
import matplotlib.pyplot as plt

from atman_magma.captum_helper import (
    CaptumMagma,
)
import numpy as np
from atman_magma.magma  import Magma
from magma.image_input import ImageInput
from  PIL import Image

from captum.attr import IntegratedGradients, InputXGradient, GuidedGradCam
from captum.attr import LayerGradCam


targets = 'Panda'
final_img = Image.open('openimages-panda.jpg')


print('loading model...')
model = Magma.from_checkpoint(
    checkpoint_path =  "./mp_rank_00_model_states.pt",
    device = 'cuda:0'
)

cmagma = CaptumMagma(magma = model)
# captum_tool = IntegratedGradients(cmagma)
# captum_tool = GuidedGradCam(cmagma, layer = cmagma.magma.lm.transformer.h[0].ln_1) #cmagma.magma.image_prefix.enc.layer4[-1].conv3) #, layer = cmagma.magma.image_prefix.enc.layer4[-1].conv3)
captum_tool = InputXGradient(cmagma)
#captum_tool = IntegratedGradients(cmagma) #! set n_steps below


cmagma.mode='text' #hack- leave it as it is - just passes below's image embeddings thru ...


label_tokens =  model.tokenizer.encode(targets)

att_combined = np.zeros((12,12))
for i in range(len(label_tokens)):

    text_prompt = f"This is a picture of a "
    if i >= 1:
        text_prompt += model.tokenizer.decode(label_tokens[:i])


    prompt = [
        ImageInput(None, pil=final_img),
        text_prompt
    ]

    embeddings = cmagma.magma.preprocess_inputs(prompt)

    attribution = captum_tool.attribute(
        embeddings,
        target=label_tokens[i],
        #n_steps = 1 #integ gradients parameters !
    )

    att = attribution[0].abs().sum(dim = 1).cpu().detach().numpy()[:144].reshape(12,12)

    att_combined += att/att.max()


fig = plt.figure()
plt.imshow(att_combined)
fig.savefig('panda-explained-captum.jpg')
print('panda-explained-captum.jpg')
