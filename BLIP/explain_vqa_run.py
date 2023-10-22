from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import json
import argparse
import pickle
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#items = glob.glob('/root/bjoern/atman_other_model/others/BLIP/test_folder/*.jpg')
#items = glob.glob('/nfs/scratch_2/bjoern/atman_other_model/openimages-mini/*/images/*.jpg')
items = glob.glob('/nfs/scratch_2/bjoern/atman_other_model/openimages-cleaned-all-classes/*/images/*.jpg')

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

def process_img(ids, run_suffix, question, targets, suppression_factor, conceptual_suppression_threshold,limit_suppressions):
    from models.blip_vqa import blip_vqa

    image_size = 480

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'
    #model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_vqa.pth'

    model = blip_vqa(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)
    cur_gpu, gpus = ids.split(',')
    cur_gpu = int(cur_gpu)
    gpus = int(gpus)

    import glob, os

    with torch.no_grad():

        #4: mask = ones except first ; .25
        #4: mask = ones except first ; .75
        #5: all ones

        #target_string="dog"
        #img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'

        #for img_url in glob.glob('/nfs/scratch_2/bjoern/atman_other_model/openimages-mini/*/images/*.jpg'):
        #target_string = img_url.split('/')[-3]
        count = 0
        for img_url in items:
            count += 1
            if count % gpus != cur_gpu:
                continue

            targets = img_url.split('/')[-3].split('(')[0]
            #target_class = 'dog'
            #question = f'Q: Is there a {targets} in the picture? '
            #targets = 'A: yes'
            question = f'What is shown in the picture?'

            for target_string in targets.split(','):
                expl_name = f'{img_url}_explanation_{run_suffix}_{target_string}.json'
                if os.path.exists(expl_name):
                    continue

                with open(f"{img_url}_similarity.pickle","rb") as dump:
                    similarities = pickle.load(dump)

                #target_string = 'green'
                image = load_demo_image(image_size=image_size, device=device,img_url=img_url)


                answer,all_factors,embedsim = model(image, question, train=False, suppression_factor=suppression_factor,
                                        conceptual_suppression_threshold=conceptual_suppression_threshold, target_string=target_string, limit_suppressions=limit_suppressions, similarities=similarities) #, limit_vision=2)
                answer_loss = answer.loss.tolist()
                #print(answer)

                with open(expl_name, 'w') as f:
                    f.write(json.dumps({'loss':answer_loss,
                                        'conceptual_suppression_threshold':conceptual_suppression_threshold,
                                        'suppression_factor':suppression_factor,
                                        'target_string':target_string,
                                        'question':question,
                                        'img':img_url,
                                        'limit_suppressions':limit_suppressions,
                                        #'factors':all_factors,
                                        #'embedsim':embedsim,
                                        'target': target_string
                                        }))

                print(img_url)
        #break

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('ids', type=str,)
parser.add_argument('suffix', type=str,)
parser.add_argument('question', type=str,)
parser.add_argument('targets', type=str,)
parser.add_argument('suppression_factor', type=float,)
parser.add_argument('conceptual_suppression_threshold', type=float,)
parser.add_argument('limit_suppressions', type=int,)

if __name__ == '__main__':

    args = parser.parse_args()
    process_img(args.ids,args.suffix,args.question,args.targets,args.suppression_factor,args.conceptual_suppression_threshold,args.limit_suppressions)
