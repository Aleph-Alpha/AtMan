from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import json
import argparse
import pickle
import glob
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#items = glob.glob('/root/bjoern/atman_other_model/others/BLIP/test_folder/*.jpg')
#items = glob.glob('/nfs/scratch_2/bjoern/atman_other_model/openimages-mini/*/images/*.jpg')
items = sorted(glob.glob('/nfs/scratch_2/bjoern/atman_other_model/openimages-cleaned-all-classes/*/images/*.jpg'))

def load_demo_image(image_size,device,img_url):
    if 'http' in img_url:
        img = Image.open(requests.get(img_url, stream=True).raw)
    else:
        img = Image.open(img_url)
    raw_image = img.convert('RGB')

    w,h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

def process_img(ids, run_suffix, question, targets, ):
    from models.blip_vqa_chefer import blip_vqa

    image_size = 480

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'
    #model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_vqa.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    image_size = 480

    model = blip_vqa(pretrained=model_url, image_size=image_size, vit='base')
    model.train()
    model = model.to(device)

    picid, gpus = ids.split(',')

    img_url = items[int(picid)]

    targets = img_url.split('/')[-3].split('(')[0]
    #target_class = 'dog'
    #question = f'Q: Is there a {targets} in the picture? '
    #targets = 'A: yes'
    question = f'What is shown in the picture?'

    for target_string in targets.split(','):
        expl_name = f'{img_url}_explanation_{run_suffix}_{target_string}.json'
        if os.path.exists(expl_name):
            print(f'skipped {expl_name}')
            continue

        #target_string = 'green'
        image = load_demo_image(image_size=image_size, device=device,img_url=img_url)


        answer3 = model.forward_chefer(image, question, target_string)

        with open(expl_name, 'w') as f:
            f.write(json.dumps({'heatmap':answer3.tolist(),
                                'target_string':target_string,
                                'question':question,
                                'img':img_url,
                                'target': target_string
                                }))


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('ids', type=str,)
parser.add_argument('suffix', type=str,)
parser.add_argument('question', type=str,)
parser.add_argument('targets', type=str,)
if __name__ == '__main__':

    args = parser.parse_args()
    process_img(args.ids,args.suffix,args.question,args.targets)
