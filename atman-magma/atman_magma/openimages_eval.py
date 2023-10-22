from tqdm import tqdm
import os
from PIL import Image
from typing import Callable

from multimodal_explain_eval.utils import check_if_a_or_an_and_get_prefix
from magma.image_input import ImageInput
from .utils import create_folder_if_does_not_exist

import torch
import numpy as np

class Metrics:
    @staticmethod
    def precision(x, y, eps = 1e-7):
        score = (x*y).sum()/(x.sum() + eps)
        return score

    @staticmethod
    def recall(x, y, eps = 1e-7):
        x = x / x.max()
        score = (x*y).sum()/(y.sum() + eps)
        return score

def run_eval(
    explainer,
    logit_parsing_fn: Callable,
    output_folder,
    metadata,
    dataloader,
    max_batch_size = 1,
    text_prompt="This is a picture of ",  ## a or an is decided later
    use_lowercase_target=True,
    auto_decide_a_or_an=True,
    progress=False,
    square_outputs: bool = False,
    num_total_explanations=None,
    prompt_explain_indices = [i for i in range(144)],
    save_configs_dir: str = None,
    save_configs_only: bool = False
):

    if progress is True and num_total_explanations is not None:
        pbar = tqdm(total=num_total_explanations)

    """
    ./result_folder
        - Cat
            - 1.jpg
            - 2.jpg
        - Dog
            - 1.jpg
            - 2.jpg
    """

    ## append a space between last word and a/an
    if auto_decide_a_or_an == True:
        if text_prompt[-1] != " ":
            text_prompt += " "


    classes = list(metadata.keys())

    d = dataloader

    if os.path.exists(output_folder) is False:
        print(f"making output_folder: {output_folder}")
        os.mkdir(output_folder)

    for i in range(len(classes)):
        output_class_dir = output_folder + "/" + classes[i]


        if os.path.exists(output_class_dir) is False:
            print(f"making folder: {output_class_dir}")
            os.mkdir(output_class_dir)


        for j in range(metadata[classes[i]]["count"]):

            filename = output_class_dir + f"/{j+1}.json"

            if os.path.exists(filename):
                print(f"{filename} already exists, skipping...")
                if progress is True:
                    pbar.update(1)
                continue

            try:
                data = d.fetch(i, j, center_crop=True, load_image = False)
            except:
                print(
                    f"an error occured while fetching from dataloader: class: {classes[i]} idx: {j+1}"
                )
                if progress is True:
                    pbar.update(1)
                continue

            input_image = ImageInput(data["image_path"])

            prompt = [
                input_image,
                text_prompt + check_if_a_or_an_and_get_prefix(word=data["label"]),
            ]

            if use_lowercase_target == True:
                target = f" {data['label'].lower()}"
            else:
                target = f" {data['label']}"

            if save_configs_dir is not None:
                folder_name = save_configs_dir + f'/{classes[i]}'
                create_folder_if_does_not_exist(folder_name)
                save_configs_as = folder_name + f'/{j+1}.json'
            else:
                save_configs_as = None

            logit_outputs = explainer.collect_logits_by_manipulating_attention(
                prompt = prompt.copy(),
                target = target,
                max_batch_size=max_batch_size,
                prompt_explain_indices = prompt_explain_indices,
                save_configs_as = save_configs_as,
                save_configs_only = save_configs_only
            )

            if save_configs_only is False:
                results = logit_parsing_fn(
                    output = logit_outputs,
                    square_outputs=square_outputs,
                )

                results.save(
                    filename = filename
                )
            torch.cuda.empty_cache()

            if progress is True:
                pbar.update(1)

    if progress is True:
        pbar.close()
    print('complete!!')
    print(f'check:\n{output_folder}')
    if save_configs_only is True:
        print('check:', save_configs_dir)


import cv2
import pandas as pd

def calculate_eval_scores_from_result_folder(
    result_folder,
    dataloader,
    metric_fn,
    output_wrapper, ## example: DeltaCrossEntropiesOutput
    output_csv_file:str = None,
    resize_to_12 = True,
    threshold =None,
    square_outputs = True
):

    result_dict = {
        'image_filename': [],
        'mask_filename': [],
        'explanation_filename': [],
        'precision_score': [],
        'recall_score': [],
        'label': [],
    }

    classes = list(dataloader.metadata.keys())
    for class_name in tqdm(classes):
        num_images = dataloader.metadata[class_name]['count']
        for idx in tqdm(range(num_images), desc = f'{class_name}', disable = True):

            # making sure the explanation filename exists :)
            x_filename = result_folder + "/" + class_name + f"/{idx+1}.json"
            assert os.path.exists(x_filename), f'Image {x_filename} does not exist'


            item = dataloader.fetch(label_idx = classes.index(class_name), idx = idx, load_image=False)
            mask_path = item['mask_path']
            image_path = item['image_path']

            y = cv2.imread(mask_path)/255.

            if resize_to_12 is True:
                y = cv2.resize(y, (12,12))

            result =  output_wrapper.from_file(
                filename = x_filename
            )


            ### consider all target tokens, not just the single tokens
            x = np.zeros((12,12))
            for target_token_idx in range(len(result.data)):
                explanation_for_single_target_token = result.show_image(image_token_start_idx = 0, target_token_idx = target_token_idx)
                if square_outputs is True:
                    explanation_for_single_target_token = explanation_for_single_target_token **2

                x += explanation_for_single_target_token

            ## thresholding
            if threshold is not None:
                x = x/x.max()
                x[x<threshold] = 0.

            score = Metrics.precision(
                x = x,
                y = y[:,:,0]
            )
            score_r = Metrics.recall(
                x = x,
                y = y[:,:,0]
            )

            result_dict['image_filename'].append(image_path)
            result_dict['mask_filename'].append(mask_path)
            result_dict['explanation_filename'].append(x_filename)
            result_dict['precision_score'].append(score)
            result_dict['recall_score'].append(score_r)
            result_dict['label'].append(item['label'])

    df = pd.DataFrame(result_dict)

    if output_csv_file is not None:
        df.to_csv(output_csv_file)
        print(f'saved: {output_csv_file}')

    return df
