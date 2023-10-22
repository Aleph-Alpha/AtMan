import fiftyone.zoo as foz
from PIL import Image
import numpy as np
import cv2

class SegmentationDataset:
    def __init__(self, 
        classes, 
        dataset_dir, 
        num_total_samples = 2000,
        width = 384, 
        height = 384,
        split = 'validation',
        dataset_name = 'open-images-animals'
    ):
        self.classes= classes
        self.num_total_samples = num_total_samples
        self.dataset_dir = dataset_dir
        self.width = width
        self.height = height

        self.dataset = foz.load_zoo_dataset(
            "open-images-v6",
            split=split,
            label_types=["segmentations"],
            classes = self.classes,
            max_samples=self.num_total_samples,
            seed=51,
            shuffle=True,
            dataset_name=dataset_name,
            dataset_dir=self.dataset_dir
        )     

    def postprocess(self, item):
        image = Image.open(item.filepath)
        
        masks = []

        crop_data = {
            'lefts': [],
            'rights': [],
            'tops': [],
            'bottoms': []
        }
        

        labels = [detection.label for detection in item.segmentations.detections]
        indices_of_labels_same_as_first_one = []

        for i in range(len(labels)):
            if labels[i] == labels[0]:
                indices_of_labels_same_as_first_one.append(i)

        # print(f'found {len(indices_of_labels_same_as_first_one)} masks for {labels[0]}: {item.filepath}\n with all labels: {labels}')

        for index in indices_of_labels_same_as_first_one:
            detection = item.segmentations.detections[index]
            # i think masks are relative, i.e. [0, 1.0] coordinates
            box = detection.bounding_box
            # not sure about the shape of the mask
            mask = detection.mask
            label = detection.label

            left, right, top, bottom = get_crop_params(
                image = image,
                bounding_box = box
            )

            crop_data['lefts'].append(left)
            crop_data['rights'].append(right)
            crop_data['tops'].append(top)
            crop_data['bottoms'].append(bottom)

            cropped_image = image.crop((left, top, right, bottom))
            
            width, height = cropped_image.size
            mask = mask.astype(np.float32)
            mask = cv2.resize(mask, (width, height))

            masks.append(mask)

        final_left = min(crop_data['lefts'])
        final_right = max(crop_data['rights'])

        final_top = min(crop_data['tops'])
        final_bottom = max(crop_data['bottoms'])

        width, height = np.array(image).shape[0], np.array(image).shape[1]
        output_mask = np.zeros((width, height)).astype(np.float32)


        for i in range(len(masks)):
            mask = masks[i]
            left = crop_data['lefts'][i]
            right = crop_data['rights'][i]
            top = crop_data['tops'][i]
            bottom = crop_data['bottoms'][i]
            output_mask[top:bottom, left:right] += mask

        cropped_image = image.crop((final_left, final_top, final_right, final_bottom))
        cropped_output_mask = output_mask[final_top:final_bottom, final_left:final_right]

        if self.height is not None and self.width is not None:
            cropped_image = cropped_image.crop((left, top, right, bottom)).resize((self.height, self.width))
            cropped_output_mask = cv2.resize(cropped_output_mask, dsize = (self.width, self.height))

        width, height = cropped_image.size
        return {
                    'image': cropped_image,
                    'mask': cropped_output_mask,
                    'label': label,
                    'filename': item.filepath,
                    'width': width,
                    'height': height
                }
        

                

def get_crop_params(image, bounding_box):
    if np.array(image).ndim == 3:
        height, width, channels = np.array(image).shape
    else:
         height, width = np.array(image).shape
        
    x0, y0, w, h = bounding_box
    
    left = int(width * x0)
    top = int(height * y0)
    right = left + int(w * width)
    bottom = top + int(h * height)
    return left, right, top, bottom