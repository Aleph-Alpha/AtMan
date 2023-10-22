from magma import Magma
from magma.image_input import ImageInput
from skimage.transform import resize
import torch
import pickle
import glob



model = Magma.from_checkpoint(
    config_path = "configs/MAGMA_v1.yml",
    checkpoint_path = "./mp_rank_00_model_states.pt",
    device = 'cuda:0'
)


#for p in glob.glob('/nfs/scratch_2/bjoern/atman_other_model/others/BLIP/test_folder/*.jpg'):
for p in glob.glob('/home/bjoern_deiseroth/bjoern/atman_other_model/openimages-cleaned-all-classes/*/images/*.jpg'):

    embeddings = model.preprocess_inputs([ImageInput(p) ])
    a_n = embeddings[0].norm(dim=1)[:, None]
    a_norm = embeddings[0] / torch.max(a_n)
    a_norm = a_norm.to('cuda:0')
    sim_mt = torch.mm(a_norm, a_norm.T).detach().cpu().numpy()

    all_result_sims = []
    for i in range(144):
        all_result_sims.append(resize(sim_mt[:,i].reshape((12,12)), (30, 30)))

    with open(f'{p}_similarity.pickle', 'wb') as handle:
        b = pickle.dump(all_result_sims, handle)
