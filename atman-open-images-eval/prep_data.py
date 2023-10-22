from pprint import pprint

from multimodal_explain_eval.data_prep import prepare_openimages_dataset
from multimodal_explain_eval.dataset import SegmentationDataset

from multimodal_explain_eval.classes import openimages_v6_segmentation_classes
from pprint import pprint
import json

# CLASSES = [
#     'Car',
#     'Bus',
#     'Cat',
#     'Dog',
#     'Pizza',
#     'Hamburger',
#     'Tiger',
#     'Tortoise',
#     'Chicken',
#     'Calculator',
#     'Clock',
#     'Toilet',
#     'Zebra'
# ]

CLASSES = openimages_v6_segmentation_classes


print(f'Preparing: {len(CLASSES)} classes')

ORIGINAL_DATASET_DIR = '/open-images-atman-eval'
CLEAN_DATASET_DIR = '/openimages-cleaned-all-classes'

all_metadata = {}

for class_name in CLASSES:

    dataset = SegmentationDataset(
        classes = [class_name],
        dataset_dir = ORIGINAL_DATASET_DIR,
        num_total_samples = 1000, # change to a large number before final run
        width = None,
        height = None,
        split = 'train',  ## change to train before final run
        dataset_name = f'open-images-animals-{class_name}'
    )

    metadata = prepare_openimages_dataset(
        segmentation_dataset = dataset,
        class_names = [class_name],
        output_folder = CLEAN_DATASET_DIR,
        num_samples_per_class = 200,
        start_clean = False,
        max_width_by_height = 1.2,
        min_width_by_height = 0.8,
        min_dim = 200
    )
    key = list(metadata.keys())[0]
    all_metadata[key] = metadata[key]

pprint(all_metadata)

with open('metadata.json', 'w') as fp:
    json.dump(all_metadata, fp, indent = 4)

count = 0

for key in all_metadata:
    count += all_metadata[key]['count']

print(f'Saved a total of: {count} images in {CLEAN_DATASET_DIR}')
