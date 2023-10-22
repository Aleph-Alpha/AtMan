import torch
import torch.nn as nn

import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2

import pandas as pd
from typing import Callable

from .utils import split_str_into_tokens

class CaptumMagma(nn.Module):
    """
    Wrapper to help with captum stuff
    """
    def __init__(
        self,
        magma,
        mode= 'image',
        text_prompt = 'This is a picture of a',
        target_token_indices = [-1]
    ):
        super().__init__()

        self.valid_modes = ['image', 'text']
        assert mode in self.valid_modes, f'Expected mode: {mode} to be one of: {self.valid_modes}'
        self.mode = mode

        self.magma = magma
        self.text_prompt = text_prompt
        self.target_token_indices = target_token_indices

    def get_logits(self, embeddings, target_token_indices = None):
        if target_token_indices is None:
            target_token_indices = self.target_token_indices


        logits  = self.magma.lm(
            inputs_embeds=embeddings,
            labels=None,
            output_hidden_states=False,
        ).logits

        return logits[:, target_token_indices,:] ## returns logits for completion tokens

    def embed_text(self, text):
        embeddings = self.magma.embed(
            [
                self.magma.tokenizer.encode(text, return_tensors = 'pt')
            ]
        )
        return embeddings

    def forward_image(self, image_tensor):

        ## if batch size is more than 1 (IntegratedGradients), then handle things differently
        if image_tensor.shape[0] > 1:
            embeddings = []

            for x in image_tensor:
                single_embedding_batch_item = self.magma.embed(
                    [
                        x.unsqueeze(0),
                        self.magma.tokenizer.encode(self.text_prompt, return_tensors="pt")
                    ]
                )
                embeddings.append(single_embedding_batch_item)
            embeddings = torch.cat(embeddings, dim = 0)
        else:
            embeddings = self.magma.embed(
                [
                    image_tensor,
                    self.magma.tokenizer.encode(self.text_prompt, return_tensors="pt")
                ]
            )

        logits = self.get_logits(embeddings = embeddings, target_token_indices = self.target_token_indices)
        # assert logits.shape[0] == 1, 'Expected batch size to be 1'
        # embeddings.shape: [1, seq, 4096]
        return logits[:,-1,:]

    def __call__(self, x):
        if self.mode == 'image':
            return self.forward_image(x)
        else:
            return self.get_logits(embeddings = x, target_token_indices = self.target_token_indices)[:,-1,:] ## return only last seq item i.e "next token"

    def load_image_as_tensor(self, filename):
        assert os.path.exists(filename), f'Expected image: {filename} to exist :('
        input_image_tensor = self.magma.transforms(Image.open(filename))

        return input_image_tensor

def collect_attributions_on_a_single_item_text(
        cmagma: CaptumMagma,
        captum_tool,
        target: str,
        prompt: str,
        device: str = 'cuda:0',
        is_integrated_gradients = True  ## true by default for convenience
    ):

    target_token_strings = split_str_into_tokens(tokenizer = cmagma.magma.tokenizer, x = target)

    results = []

    for i in range(len(target_token_strings)):
        target_token_str = target_token_strings[i]
        target_tokenized = cmagma.magma.tokenizer.encode(target_token_str)

        if i > 0:
            prompt += target_token_strings[i-1]

        assert len(target_tokenized) == 1
        target_token_id = target_tokenized[0]

        embeddings = cmagma.embed_text(prompt).to(device)

        if is_integrated_gradients == True:
            attribution = captum_tool.attribute(embeddings, target=target_token_id, n_steps = 10)
        else:
            attribution = captum_tool.attribute(embeddings, target=target_token_id)

        data = {
            'attribution': attribution,
            'target_token_str':target_token_str,
            'target_token_id': target_token_id
        }
        results.append(data)

    return results

def collect_attributions_on_a_single_item(
        image_path: str,
        cmagma: CaptumMagma,
        captum_tool,
        target: str,
        text_prompt = 'This is a picture of a',
        is_integrated_gradients = True  ## true by default for convenience
    ):
    cmagma.text_prompt = text_prompt

    image_tensor = cmagma.load_image_as_tensor(image_path)

    target_token_strings = split_str_into_tokens(tokenizer = cmagma.magma.tokenizer, x = target)

    results = []

    for i in range(len(target_token_strings)):
        target_token_str = target_token_strings[i]
        target_tokenized = cmagma.magma.tokenizer.encode(target_token_str)

        if i > 0:
            cmagma.text_prompt += target_token_strings[i-1]

        assert len(target_tokenized) == 1
        target_token_id = target_tokenized[0]
        if is_integrated_gradients == True:
            attribution = captum_tool.attribute(image_tensor, target=target_token_id, n_steps = 10)
        else:
            attribution = captum_tool.attribute(image_tensor, target=target_token_id)

        data = {
            'attribution': attribution,
            'target_token_str':target_token_str,
            'target_token_id': target_token_id
        }
        results.append(data)

    return results


def parse_results_over_all_tokens(results, square_outputs = True, divide_by_max = True):

    attributions = []

    for i in range(len(results)):
        attr = results[i]['attribution'].cpu().detach()[0].permute(1,2,0).mean(-1)
        if square_outputs is True:
            attr = attr**2
        if divide_by_max is True:
            attr = attr/attr.max()

        attributions.append(attr.numpy())

    return sum(attributions)/len(attributions)


def run_eval_with_captum_tool(
    cmagma,
    captum_tool,
    output_folder,
    metadata,
    dataloader,
    text_prompt="This is a picture of ",  ## a or an is decided later
    use_lowercase_target=True,
    auto_decide_a_or_an=True,
    progress=False,
    square_outputs: bool = False,
    num_total_explanations=None,
    divide_by_max = False,
    is_integrated_gradients = True
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

            filename = output_class_dir + f"/{j+1}.npy"

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


            if use_lowercase_target == True:
                target = f" {data['label'].lower()}"
            else:
                target = f" {data['label']}"

            results = collect_attributions_on_a_single_item(
                image_path = data['image_path'],
                cmagma = cmagma,
                captum_tool = captum_tool,
                target = target,
                text_prompt = text_prompt,
                is_integrated_gradients = is_integrated_gradients
            )

            ## heatmap_numpy is a numpy array with values between 0 and 1 if divide_by_max == True
            heatmap_numpy = parse_results_over_all_tokens(
                results = results,
                square_outputs=square_outputs,
                divide_by_max=divide_by_max
            )
            np.save(filename, heatmap_numpy)
            print(f'saved: {filename}')
            torch.cuda.empty_cache()
            if progress is True:
                pbar.update(1)

    if progress is True:
        pbar.close()
    print('complete!!')
    print(f'check:\n{output_folder}')


def calculate_score_for_output_folder(
    output_folder: str,
    metadata,
    dataloader,
    output_csv_file: str,
    metric_fn: Callable,
    progress=True,
    num_total_explanations=None,
    mask_size = (384,384),
):

    result_dict = {
        'image_filename': [],
        'mask_filename': [],
        'explanation_filename': [],
        'precision_score': [],
        'label': [],
    }

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
    classes = list(metadata.keys())

    d = dataloader
    for i in range(len(classes)):
        output_class_dir = output_folder + "/" + classes[i]

        for j in range(metadata[classes[i]]["count"]):

            try:
                data = d.fetch(i, j, center_crop=True, load_image = False)
            except:
                print(
                    f"an error occured while fetching from dataloader: class: {classes[i]} idx: {j+1}"
                )
                if progress is True:
                    pbar.update(1)
                continue

            explanation_path = output_class_dir + f"/{j+1}.npy"
            mask_path = data['mask_path']
            image_path = data['image_path']

            assert os.path.exists(explanation_path), f'Expected explanation path: {explanation_path} to exist :('
            explanation = np.array(np.load(explanation_path))
            mask = np.array(Image.open(mask_path).resize(mask_size))/255.

            score = metric_fn(x= explanation, y=mask)

            result_dict['image_filename'].append(image_path)
            result_dict['mask_filename'].append(mask_path)
            result_dict['explanation_filename'].append(explanation_path)
            result_dict['precision_score'].append(score)
            result_dict['label'].append(data['label'])

            if progress is True:
                pbar.update(1)

    df = pd.DataFrame(result_dict)
    df.to_csv(output_csv_file)
    print(f'saved: {output_csv_file}')

    return df
