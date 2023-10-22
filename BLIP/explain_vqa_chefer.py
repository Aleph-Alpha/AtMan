from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import json

from multiprocessing import Pool
import glob, os


run_suffix = 'chefer3'
num_gpus = 2

#question = 'A picture of a '
question = 'What is shown in the picture?'
#targets ='dog,cat'
targets=''

#question = ''
#targets = ''


#question = 'What is shown in the picture?'
#targets ='Dog,Cat'


items = glob.glob('/nfs/scratch_2/bjoern/atman_other_model/openimages-cleaned-all-classes/*/images/*.jpg')

def process_x(x):
    # if targets == '':
    #     targets = items[x].split('/')[-3]
    # else:
    #     targets += ","+items[x].split('/')[-3]

    os.system(f'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={x%num_gpus} python explain_vqa_run_chefer.py "{x},{num_gpus}" {run_suffix} "{question}" "{targets}"')


import math

#items = items[:4]

with Pool(num_gpus) as p:
    p.map(process_x , range(len(items)))

#process_x(0)
