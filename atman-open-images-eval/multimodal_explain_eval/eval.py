import os
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import torch


from .dataloader import DataLoader
from .utils import (
    result_to_segmentation_mask,
    check_if_a_or_an_and_get_prefix,
    dict_to_json,
)

from generator import TaskExplanation, TaskEvaluation, TaskCompletion


def get_embedding(
    generator_context,
    text="hello world",
):

    with generator_context as generator:
        token_ids = torch.tensor(generator.tokenizer.encode(text)).cuda()
        emb = generator.transformer.transformer.embeddings.word_embeddings(token_ids)

    return emb


def get_logprob(
    generator,
    text_prompt,
    pil_image,
    label,
    use_lowercase_target=True,
    auto_decide_a_or_an=True,
    also_collect_completion=True,
):

    if auto_decide_a_or_an == True:
        if text_prompt[-1] != " ":
            text_prompt += " "

    prompt = [pil_image, text_prompt + check_if_a_or_an_and_get_prefix(word=label)]

    if use_lowercase_target == True:
        target = f" {label.lower()}"
    else:
        target = f" {label}"

    # print(prompt, target)

    task = TaskEvaluation(prompt=prompt, completion_expected=target)

    output = generator.process([task])
    data = {
        "original_completion": None,
        "logprob_on_target": output[0]["result"]["log_probability"],
    }

    if also_collect_completion is not None:
        task = TaskCompletion(prompt=prompt, maximum_tokens=5)

        output = generator.process([task])

        data["original_completion"] = output[0]["completions"][0]["completion"]

    return data


def run_eval(
    generator,
    result_folder,
    metadata,
    dataloader: DataLoader,
    text_prompt="This is a picture of ",  ## a or an is decided later
    suppression_factor=0.1,
    conceptual_suppression_threshold=0.5,
    max_batch_size=32,
    use_lowercase_target=True,
    auto_decide_a_or_an=True,
    progress=False,
    raw_jsons_folder: str = None,
    normalize: bool = False,
    square_outputs: bool = False,
    num_total_explanations=None,
    logger = None
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

    RESULTS = {}

    classes = list(metadata.keys())

    d = dataloader

    if os.path.exists(result_folder) is False:
        print(f"making folder: {result_folder}")
        os.mkdir(result_folder)

    if os.path.exists(raw_jsons_folder) is False:
        print(f"making folder: {raw_jsons_folder}")
        os.mkdir(raw_jsons_folder)

    for i in range(len(classes)):
        label_dir = result_folder + "/" + classes[i]
        json_dir = raw_jsons_folder + "/" + classes[i]

        RESULTS[classes[i]] = {"folder": label_dir, "count": 0}

        if os.path.exists(label_dir) is False:
            print(f"making folder: {label_dir}")
            os.mkdir(label_dir)

        if os.path.exists(json_dir) is False and raw_jsons_folder is not None:
            print(f"making folder: {json_dir}")
            os.mkdir(json_dir)

        for j in range(metadata[classes[i]]["count"]):

            # 1 indexed filenames (starts with 1.jpg)
            filename = label_dir + f"/{j+1}.jpg"
            json_filename = json_dir + f"/{j+1}.json"

            if os.path.exists(filename):
                print(f"{filename} already exists, skipping...")
                RESULTS[classes[i]]["count"] += 1
                pbar.update(1)
                continue

            try:
                data = d.fetch(i, j, center_crop=True)
            except:
                print(
                    f"an error occured while fetching from dataloader: class: {classes[i]} idx: {j+1}"
                )

                if logger is not None:
                    logger.info(
                    f"an error occured while fetching from dataloader: class: {classes[i]} idx: {j+1}"
                )
                pbar.update(1)
                continue

            input_image = Image.fromarray(data["image"])

            prompt = [
                input_image,
                text_prompt + check_if_a_or_an_and_get_prefix(word=data["label"]),
            ]

            if use_lowercase_target == True:
                target = f" {data['label'].lower()}"
            else:
                target = f" {data['label']}"

            task = TaskExplanation(
                prompt=prompt,
                target=target,
                suppression_factor=suppression_factor,
                conceptual_suppression_threshold=conceptual_suppression_threshold,
                normalize=normalize,
                square_outputs=square_outputs,
            )

            result = generator.process([task], max_batch_size=max_batch_size)[0]

            mask = result_to_segmentation_mask(
                result,
                target_token_idx=[i for i in range(len(result["result"]))],
                width=384,
                height=384,
            )

            cv2.imwrite(filename, (mask * 255).astype(np.uint8))
            if logger is not None:
                logger.info(f'saved: {filename}')
            else:
                print(f"saved: {filename}")

            if raw_jsons_folder is not None:
                result["explanation_filename"] = filename

                dict_to_json(dictionary=result, filename=json_filename)

                if logger is not None:
                    logger.info(f"saved json: {json_filename}")
                else:
                    print(f"saved json: {json_filename}")

            RESULTS[classes[i]]["count"] += 1
            pbar.update(1)
    pbar.close()
    return RESULTS
