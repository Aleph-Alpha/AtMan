from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import json

from multiprocessing import Pool
import glob, os


run_suffix = 'run2'
num_gpus = 2

suppression_factor=.05
conceptual_suppression_threshold=.10
limit_suppression=150

#question = 'A picture of a '
question = 'What is shown in the picture?'
#targets ='dog,cat'
targets=''

#question = ''
#targets = ''


#question = 'What is shown in the picture?'
#targets ='Dog,Cat'


def process_x(x):
    # if targets == '':
    #     targets = items[x].split('/')[-3]
    # else:
    #     targets += ","+items[x].split('/')[-3]

    os.system(f'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={x%num_gpus} python explain_vqa_run.py "{x},{num_gpus}" {run_suffix} "{question}" "{targets}" {suppression_factor} {conceptual_suppression_threshold} {limit_suppression}')


import math

with Pool(num_gpus) as p:
    p.map(process_x , range(num_gpus))

#process_x(0)
