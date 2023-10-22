from .dataset import SegmentationDataset
import os
from tqdm import tqdm
import numpy as np 
import cv2
import shutil

from .utils import check_if_a_or_an_and_get_prefix

def get_filenames_in_a_folder(folder: str):
    """
    returns the list of paths to all the files in a given folder
    """

    files = os.listdir(folder)
    num_files = len(files)
    """
    - class A
        - images
            - 1.jpg
            - 2.jpg
        - masks
            - 1.jpg
            - 2.jpg

    - class B
        - images
            - 1.jpg
            - 2.jpg
        - masks
            - 1.jpg
            - 2.jpg
    """

    filenames_sorted = [
        f"{folder}/{x+1}.jpg" for x in range(num_files)
    ]
    return filenames_sorted

def prepare_openimages_dataset(
    segmentation_dataset: SegmentationDataset,
    class_names: list,
    output_folder = './openimages_cleaned',
    num_samples_per_class = 100,
    start_clean = False,
    max_width_by_height = 1.2,
    min_width_by_height = 0.8,
    min_dim = 200
):

    if os.path.exists(output_folder):
        print(f'folder: {output_folder} already exists with folders:\n {get_filenames_in_a_folder(output_folder)}')
        if start_clean is True:
            print('starting clean, deleting existing folder...')
            shutil.rmtree(output_folder)
            os.mkdir(output_folder)
    else:
        print(f'making folder: {output_folder}')
        os.mkdir(output_folder)
    
    metadata = {}

    for name in class_names:
        metadata[name] = {
            'count': 0,
            'prefix': check_if_a_or_an_and_get_prefix(word = name),
            'folder': {
                'images': None,
                'masks': None
            }
        }

    """
    output_folder/
        dog/
            images/
                1.jpg
                2.jpg
            masks/
                1.jpg
                2.jpg
        cat/
            images/
                1.jpg
                2.jpg
            masks/
                1.jpg
                2.jpg
    """

    for item in tqdm(segmentation_dataset.dataset, desc = 'preparing data:'):
        stuff = segmentation_dataset.postprocess(item)
        label = stuff['label']

        if label in list(metadata.keys()):

            label_dir = output_folder + '/' + label
            image_dir = label_dir + '/images'
            mask_dir = label_dir + '/masks'

            if os.path.exists(label_dir) is False:
                print(f'making folder: {label_dir}')
                os.mkdir(label_dir)

            if os.path.exists(image_dir) is False:
                print(f'making folder: {image_dir}')
                os.mkdir(image_dir)

            if os.path.exists(mask_dir) is False:
                print(f'making folder: {mask_dir}')
                os.mkdir(mask_dir)

            ## I know this is redundant, but we'll run it only ones, so this worrks :)
            metadata[label]['folder']['images'] = image_dir
            metadata[label]['folder']['masks'] = mask_dir

            if metadata[label]['count'] >= num_samples_per_class:
                continue

            image = stuff['image']
            mask = stuff['mask']

            width, height = stuff['width'], stuff['height']

            ratio = width/height

            minimum_dim_on_image = min([width, height])

            if min_width_by_height < ratio < max_width_by_height and minimum_dim_on_image >= min_dim:
                metadata[label]['count'] = metadata[label]['count'] + 1
                image_path = image_dir + f"/{metadata[label]['count']}.jpg"
                mask_path = mask_dir + f"/{metadata[label]['count']}.jpg"
                # print(f'filename {item.filepath}, label dir: {label_dir}, image_path: {image_path}')

                if os.path.exists(image_path) is False:
                    image.save(image_path)
                    # print(f'saving: {image_path}')
                else:
                    print(f'already exists: {image_path}')
                cv2.imwrite(mask_path, (mask*255).astype(np.uint8))
            else:
                # print(f'rejecting: {label} because bad ratio: {ratio} or shape lower than min dim: {width, height}')
                continue

    for label, data in metadata.items():

        try:
            assert data['count'] == num_samples_per_class, f"Expected {num_samples_per_class} samples for {label} but got {data['count']}"
        except AssertionError:
            print('WARNING:', f"Expected {num_samples_per_class} samples for {label} but got {data['count']}")

        try:
            assert len(get_filenames_in_a_folder(data['folder']['images'])) == num_samples_per_class, f"Got {len(get_filenames_in_a_folder(data['folder']['images']))} images but expected {num_samples_per_class} images in folder: {get_filenames_in_a_folder(data['folder']['images'])}"
        except AssertionError:
            print('WARNING:', f"Got {len(get_filenames_in_a_folder(data['folder']['images']))} images but expected {num_samples_per_class} images in folder")
        
        try:
            assert len(get_filenames_in_a_folder(data['folder']['masks'])) == num_samples_per_class, f"Got {len(get_filenames_in_a_folder(data['folder']['masks']))} masks but expected {num_samples_per_class} masks in folder: {get_filenames_in_a_folder(data['folder']['masks'])}"
        except AssertionError:
            print('WARNING:', f"Got {len(get_filenames_in_a_folder(data['folder']['masks']))} masks but expected {num_samples_per_class} masks in folder")

    return metadata