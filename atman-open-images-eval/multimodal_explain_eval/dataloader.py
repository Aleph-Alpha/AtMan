from PIL import Image
import cv2
from .data_prep import get_filenames_in_a_folder
import os 

def center_crop_image(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped

    source: https://gist.github.com/Nannigalaxy/35dd1d0722f29672e68b700bc5d44767
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img


class DataLoader():
    def __init__(self, metadata, load_image_on_getitem = False):
        
        """
        metadata example:
        metadata = {'Cat': {'count': 10,
         'folder': {'images': './openimages_cleaned/Cat/images',
                    'masks': './openimages_cleaned/Cat/masks'}},
         'Dog': {'count': 10,
                 'folder': {'images': './openimages_cleaned/Dog/images',
                            'masks': './openimages_cleaned/Dog/masks'}}
        }
        """
        self.load_image_on_getitem = load_image_on_getitem
        self.metadata = metadata
        self.labels = list(metadata.keys())
        
        self.image_paths = {}
        self.mask_paths = {}
        
        for l in self.labels:
            self.image_paths[l] = get_filenames_in_a_folder(self.metadata[l]['folder']['images'])
            self.mask_paths[l] = get_filenames_in_a_folder(self.metadata[l]['folder']['masks'])

        self.all_item_indices = self.prepare_stuff_for_iter(metadata=self.metadata)
        print(f'dataloader contains a total of {len(self.all_item_indices)} indices')
        
    def fetch(self, label_idx = 0, idx = 0, center_crop = False, load_image = True):
        assert label_idx < len(self.labels)
        assert idx < len(self.image_paths[self.labels[label_idx]]), f'Got idx: {idx} but the highest was {len(self.image_paths[self.labels[label_idx]])}'
        
        image_path = self.image_paths[self.labels[label_idx]][idx]
        assert os.path.exists(image_path), f'Expected image to exist in:{image_path}'
        mask_path = self.mask_paths[self.labels[label_idx]][idx]
    
        if load_image == True:
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        else:
            image = None

        mask = cv2.imread(mask_path)
        if image is not None:
            h, w, c = image.shape
        else:
            h, w, c = mask.shape

        if center_crop is True:
            min_dim = min([w,h])
            if image is not None:
                image  = center_crop_image(image, dim  = (min_dim, min_dim))
            mask = center_crop_image(mask, dim  = (min_dim,min_dim))
        
        return {
            'image': image,
            'mask': mask,
            'image_path': image_path,
            "mask_path": mask_path,
            'label': self.labels[label_idx]
        }

    def prepare_stuff_for_iter(self, metadata):
        classes = list(metadata.keys())
        all_item_indices = []

        for i in range(len(classes)):
            for j in range(metadata[classes[i]]["count"]):
                all_item_indices.append(
                    {
                        'label_idx': i,
                        'item_idx': j
                    }
                )

        return all_item_indices

    def __len__(self):
        return len(self.all_item_indices)

    def __getitem__(self, idx):
        return self.fetch(
            label_idx= self.all_item_indices[idx]['label_idx'],
            idx = self.all_item_indices[idx]['item_idx'],
            load_image=self.load_image_on_getitem
        )
        