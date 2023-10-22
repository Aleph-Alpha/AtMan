import matplotlib.pyplot as plt
import cv2
import numpy as np
import json


def load_json_as_dict(filename: str):
    json1_file = open(filename)
    json1_str = json1_file.read()
    d = json.loads(json1_str)
    return d


def check_if_a_or_an_and_get_prefix(word: str):
    vowels = ["a", "e", "i", "o", "u"]
    if word[0].lower() in vowels:
        return "an"
    else:
        return "a"


def normalize(x, eps=1e-8):
    """
    values are shifted and rescaled so that they end up ranging between 0 and 1
    """
    normalized_array = (x - x.min()) / (x.max() - x.min()) + eps
    return normalized_array


def get_superimposed_heatmap(image, heatmap, colormap=cv2.COLORMAP_VIRIDIS):
    colored_heatmap = cv2.applyColorMap(heatmap, colormap)
    return 0.5 * image / 255.0 + 0.5 * (colored_heatmap / 255.0)


def result_to_segmentation_mask(
    result, target_token_idx: list, width=None, height=None
):
    values = []
    dict_values = []
    for idx in target_token_idx:

        try:
            dict_values.append(
                result["result"][idx]["explanations"][:144]
            )  ## make it to "explanations" for the API
            values.append(
                np.array([x["importance"] for x in dict_values[-1]]).reshape(12, 12)
            )  ## value -> "importance" for the API
        except KeyError:
            dict_values.append(result["result"][idx]["explanation"][:144])
            values.append(
                np.array([x["value"] for x in dict_values[-1]]).reshape(12, 12)
            )

    for v in values:
        v[0, 0] = v.mean()
        v[-1, -1] = v.mean()

    ## interpolation = cv2.INTER_NEAREST

    values = sum(values) / len(values)

    if width is not None and height is not None:
        values = cv2.resize(values, (width, height), interpolation=cv2.INTER_NEAREST)
    return values


def visualize_result(
    prompt_image_pil,
    prompt,
    target,
    result,
    target_token_idx,
    suppression_factor,
    conceptual_suppression_threshold,
    fontsize=20,
    colormap=cv2.COLORMAP_VIRIDIS,
    sigma=None,
):
    dict_values = []
    values = []
    for idx in target_token_idx:
        dict_values.append(result["result"][idx]["explanations"][:144])
        values.append(
            np.array([x["importance"] for x in dict_values[-1]]).reshape(12, 12)
        )

    for v in values:
        v[0, 0] = v.mean()
        v[-1, -1] = v.mean()

    ## interpolation = cv2.INTER_NEAREST
    values = cv2.resize(
        sum(values) / len(values), (384, 384), interpolation=cv2.INTER_NEAREST
    ).reshape(384, 384, 1)

    if sigma is not None:
        values = values**sigma

    out = get_superimposed_heatmap(
        image=cv2.resize(
            np.array(prompt_image_pil),
            dsize=(384, 384),
        ),
        heatmap=(normalize(values) * 255).astype(np.uint8),
        colormap=colormap,
    )

    input_image = cv2.resize(
        np.array(prompt_image_pil),
        dsize=(384, 384),
    )

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(30, 9))
    fig.suptitle(f"PROMPT: <IMAGE> {prompt[-1]}\ntarget: {target}", fontsize=fontsize)

    ax[0].set_title("input image")
    ax[0].imshow(input_image)

    ax[1].set_title(
        f"target token: {''.join([result['result'][idx]['target_token_str'] for idx in target_token_idx])} \nsuppression_factor: {suppression_factor}\nconceptual_suppression_threshold: {conceptual_suppression_threshold}",
        fontsize=fontsize,
    )
    ax[1].imshow(out)

    im = ax[2].imshow(values)
    ax[2].set_title(f"relative importance", fontsize=fontsize)

    im = ax[3].imshow(values, vmin=0, vmax=1)
    ax[3].set_title(
        f"non normalized\n max:{round(values.max(), 5)}\n min:{round(values.min(), 5)}",
        fontsize=fontsize,
    )

    fig.colorbar(im, ax=ax)

    return input_image, values, out


import base64
from io import BytesIO


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    buffer = buffered.getvalue()
    return buffer


def decode(jpg_as_text):
    pil_image = Image.open(BytesIO(base64.b64decode(jpg_as_text)))
    return pil_image


def dict_to_json(dictionary, filename):
    with open(filename, "w") as fp:
        json.dump(dictionary, fp, indent=4)


def calculate_time(
    num_classes,
    num_images_per_class,
    num_conceptual_suppression_thresholds,
    num_seconds_per_sample,
):

    t = round((
        num_classes*num_images_per_class*num_conceptual_suppression_thresholds*num_seconds_per_sample
    )/3600, 4)

    print(
        f'''\nnum_classes: {num_classes}
num_images_per_class: {num_images_per_class}
num_conceptual_suppression_thresholds: {num_conceptual_suppression_thresholds}
num_seconds_per_sample: {num_seconds_per_sample}

time estimate: {t} hours
---------------------------------------------------------------\n
'''
    )
    return t 


from typing import Callable

def parse_results_for_single_image(
    folder: str,
    class_name: str,
    idx: int,
    dataloader,
    metric_fn: Callable,
    minimum_image_size: int = 384,
    min_w_by_h: float = 0.9,
    max_w_by_h: float = 1.1,
    threshold: float = 0.1,
    resize_to_12: bool = True,
    result_dict: dict = None
):
    '''
    result_dict looks like this:
    SCORES_FOR_EACH_FILENAME = {
        'image_filename': [],
        'mask_filename': [],
        'explanation_filename': [],
        'precision_score': [],
        'label': [],
    }
    '''
    classes = list(dataloader.metadata.keys())

    data = dataloader.fetch(classes.index(class_name), idx, center_crop=True, load_image = False)
    y = data["mask"]
    y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
    y = y / y.max()

    (
        w,
        h,
    ) = y.shape

    min_dim = min([w, h])

    ratio = w / h

    ## should have nice aspect ratio and big size, only then do we calculate precision score, else we return None
    if min_dim > minimum_image_size and min_w_by_h < ratio < max_w_by_h:

        ## its idx + 1 because the filenames were 1 indexed
        explanation_path = folder + "/" + class_name + f"/{idx+1}.jpg"
        x = cv2.imread(explanation_path)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x = cv2.resize(x, dsize=(w, h))
        x = x / (x.max() + 1e-8)

        if threshold is not None:
            x[x < threshold] = 0.0

        ## resize to 12x12 grids
        if resize_to_12 is True:
            x = cv2.resize(x, (12, 12))
            y = cv2.resize(y, (12, 12))
        score = metric_fn(x, y)
        
        if result_dict is not None:
            ## keeping scores for each image file
            result_dict['image_filename'].append(
                data['image_path']
            )
            result_dict['mask_filename'].append(
                data['mask_path']
            )
            result_dict['explanation_filename'].append(
                explanation_path
            )
            result_dict['precision_score'].append(
                score
            )
            
            result_dict['label'].append(
                class_name
            )   

        return score
    else:
        return None

from tqdm import tqdm
import os 
import pandas as pd

def calculate_eval_scores_from_result_folder(
    folder,
    output_csv_file,
    dataloader,
    metric_fn,
    minimum_image_size = 384,
    min_w_by_h = 0.1,
    max_w_by_h = 1.5,
    threshold = 0.1,
    resize_to_12 = True,
):

    result_dict = {
        'image_filename': [],
        'mask_filename': [],
        'explanation_filename': [],
        'precision_score': [],
        'label': [],
    }


    for class_name in tqdm(list(dataloader.metadata.keys())):
        num_images = dataloader.metadata[class_name]['count']
        for idx in tqdm(range(num_images), desc = f'{class_name}', disable = True):
            # calculate score
            try:
                ## making sure the explanation filename exists :)
                filename = folder + "/" + class_name + f"/{idx+1}.jpg"
                assert os.path.exists(filename), f'Image {filename} does not exist'

                score = parse_results_for_single_image(
                    result_dict = result_dict,
                    folder = folder,
                    class_name = class_name,
                    idx = idx,
                    dataloader =dataloader,
                    metric_fn = metric_fn,
                    minimum_image_size = minimum_image_size,
                    min_w_by_h = min_w_by_h,
                    max_w_by_h = max_w_by_h,
                    threshold = threshold,
                    resize_to_12 = resize_to_12
                )
            except:
                print(f'An error occured for class_name: {class_name} idx: {idx}')

    df = pd.DataFrame(result_dict)
    df.to_csv(output_csv_file)
    print(f'saved: {output_csv_file}')

    return df