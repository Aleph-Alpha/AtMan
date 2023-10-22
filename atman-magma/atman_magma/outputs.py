import json
import numpy as np
from .utils import load_json_as_dict

class DeltaCrossEntropiesOutput:
    def __init__(
        self,
        data: dict
    ):
        self.data = data
        self.num_tokens_in_image = 144

    def save(self, filename: str):
        with open(filename, 'w') as fp:
            json.dump(self.data, fp, indent = 4)
        print(f'saved: {filename}')

    def get_text_heatmap(self, target_token_idx: int, square_outputs = False):
        selected_item = self.data[target_token_idx]
        target_token_id = selected_item['target_token_id']
        heatmap = np.array(
                [
                x['value'] for x in selected_item['explanation']
            ]
        )

        if square_outputs is True:
            heatmap = heatmap**2

        return {
            'token_id':target_token_id,
            'heatmap': heatmap
        }

    @classmethod
    def from_file(cls, filename: str):
        data = load_json_as_dict(
            filename = filename
        )

        return cls(data = data)

        

    def show_image(self, image_token_start_idx, target_token_idx: int):
        x = self.data[target_token_idx]
        image_explanation_values = [
            x['value'] for x in x['explanation'][image_token_start_idx:image_token_start_idx+self.num_tokens_in_image]
        ]
        heatmap = np.array(image_explanation_values).reshape(12,12)
        
        return heatmap
    
class DeltaLogitsOutput:
    def __init__(
        self,
        data: dict
    ):
        self.data = data
        self.num_tokens_in_image = 144

    def save(self, filename: str):
        with open(filename, 'w') as fp:
            json.dump(self.data, fp, indent = 4)
        print(f'saved: {filename}')

    def get_text_heatmap(self, target_token_idx: int, square_outputs = False):
        selected_item = self.data[target_token_idx]
        target_token_id = selected_item['target_token_id']
        heatmap = np.array(
                [
                x['value'] for x in selected_item['explanation']
            ]
        )

        if square_outputs is True:
            heatmap = heatmap**2

        return {
            'token_id':target_token_id,
            'heatmap': heatmap
        }

    @classmethod
    def from_file(cls, filename: str):
        data = load_json_as_dict(
            filename = filename
        )

        return cls(data = data)

        

    def show_image(self, image_token_start_idx, target_token_idx: int):
        x = self.data[target_token_idx]
        image_explanation_values = [
            x['value'] for x in x['explanation'][image_token_start_idx:image_token_start_idx+self.num_tokens_in_image]
        ]
        heatmap = np.array(image_explanation_values).reshape(12,12)
        
        return heatmap
