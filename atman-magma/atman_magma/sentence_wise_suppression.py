from typing import Callable, List
import numpy as np


def split_list_based_on_indices(list_to_split, split_indices):
    token_ids = np.array(list_to_split)
    split_token_ids = np.split(token_ids, indices_or_sections= split_indices)

    return [x.tolist() for x in split_token_ids]

def search_multi_token_delimiter(token_ids, delimiter):
    """Splits a list of token IDs based on a multi-token (or single token) delimiter
    Note that it includes the delimiter itself at the end of each chunk
    token_ids: list of token ids in prompt
    delimiter: list of toke ids which represent the delimiter

    For example:
    ```python
    token_ids = [0,0,1,2,3,4,5,3,4,5,6,7,8,3,4,9,9,9,2,1,3,56,2]
    delimiter = [3,4]  ## multi token delimiter
    search_multi_token_delimiter(token_ids, delimiter)
    ```
    >>> [[0, 0, 1, 2, 3, 4], [5, 3, 4], [5, 6, 7, 8, 3, 4], [9, 9, 9, 2, 1, 3, 56, 2]]
    """
    len_delimiter = len(delimiter)

    split_indices = []
    for i in range(0, len(token_ids)):
        if token_ids[i:i+len_delimiter] == delimiter:
            split_indices.append(i+len_delimiter)

    return split_indices


class SentenceWiseSuppression:
    def __init__(self, prompt: str, delimiters: List[str], tokenizer_encode_fn: Callable, tokenizer_decode_fn: Callable):
        """
        Provides config for suppressing a prompt one sentence at a time.
        The delimiter_token can be used to specify how a sentence ends, 
        like "\n" (newline) or "." (fullstop)
        """
        assert isinstance(prompt, str)
        assert isinstance(delimiters, list)


        self.tokenizer_encode_fn = tokenizer_encode_fn
        self.tokenizer_decode_fn = tokenizer_decode_fn
        self.delimiters = delimiters

        self.prompt_token_ids = self.tokenizer_encode_fn(prompt)
        self.delimiter_token_ids_list = [
            self.tokenizer_encode_fn(d) for d in self.delimiters
        ]

        
    def get_split_prompt_token_id_indices_based_on_delimiter(self, delimiter_token_ids_list, prompt_token_ids, return_indices = True):
        

        all_split_indices = []
        for delimiter_token_ids in delimiter_token_ids_list:
            all_split_indices.extend(
                search_multi_token_delimiter(prompt_token_ids, delimiter_token_ids)
            )

        all_split_indices.sort()

        data = split_list_based_on_indices(list_to_split=prompt_token_ids, split_indices=all_split_indices)
        
        start = 0
        if return_indices is True:
            chunk_indices = []
            
            for prompt_chunk_ids_list in data:
                chunk_indices.append(
                    [i for i in range(start, start + len(prompt_chunk_ids_list))]
                )
                start += len(prompt_chunk_ids_list)
        
            return chunk_indices
        else:
            return data
        
    def get_split_prompt(self):
        split_token_ids = self.get_split_prompt_token_id_indices_based_on_delimiter(
            delimiter_token_ids_list = self.delimiter_token_ids_list,
            prompt_token_ids = self.prompt_token_ids,
            return_indices=False
        )
        
        split_prompt = [
            self.tokenizer_decode_fn(chunk) for chunk in split_token_ids
        ]
        return split_prompt
            
    def get_config(self, suppression_factor: float):

        '''
        custom_config = [
            ## should always be present
            {
                "suppression_token_index": [-1], 
                "suppression_factor": [1.0]
            },
            ## suppress token indices 0,1,2,3 together
            {
                "suppression_token_index": [0, 1, 2, 3], 
                "suppression_factor": [0.1, 0.1, 0.1, 0.1]
            },
            ## suppress token indices 4,5,6,7 together
            {
                "suppression_token_index": [4, 5, 6, 7], 
                "suppression_factor": [0.1, 0.1, 0.1, 0.1]
            },
        ]
        '''
        prompt_chunk_indices =  self.get_split_prompt_token_id_indices_based_on_delimiter(
            delimiter_token_ids_list = self.delimiter_token_ids_list,
            prompt_token_ids = self.prompt_token_ids,
            return_indices=True
        )
        
        custom_config = [
            {
                "suppression_token_index": [-1], 
                "suppression_factor": [1.0]
            }
        ]
        
        for chunk_indices in prompt_chunk_indices:
            custom_config.append(
                {
                    "suppression_token_index": chunk_indices,
                    "suppression_factor": [suppression_factor for i in chunk_indices]
                }
            )
            
        return custom_config
            
        
        