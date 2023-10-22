import torch.nn as nn
import torch
import copy

from .utils import chunks, get_output_logits_from_embeddings, dict_to_json
from .conceptual_suppression import ConceptualSuppression

class Explainer:
    def __init__(self, model: nn.Module, device, tokenizer, suppression_factor = 0.1, conceptual_suppression_threshold = 0.6,
                 modify_suppression_factor_based_on_cossim = True,
                 multiplicative=True,
                 do_log = False,
                 layers=None,
                 manipulate_attn_scores_after_scaling: bool = False):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.layers = layers
        self.suppression_factor = suppression_factor
        self.conceptual_suppression_threshold = conceptual_suppression_threshold
        self.modify_suppression_factor_based_on_cossim = modify_suppression_factor_based_on_cossim
        self.manipulate_attn_scores_after_scaling = manipulate_attn_scores_after_scaling

        self.model.suppression_factor = suppression_factor
        if conceptual_suppression_threshold is not None:
            assert 0 <= conceptual_suppression_threshold <= 1., f'Expected conceptual_suppression_threshold to be between 0 and 1, but got: {conceptual_suppression_threshold}'

        self.model.conceptual_suppression_threshold = conceptual_suppression_threshold

    def collect_logits_by_manipulating_attention(self, prompt: list, target: str, prompt_explain_indices: list = None, max_batch_size: int = 1, configs: dict = None, save_configs_as: str = None, save_configs_only: bool = False):
        """Runs forward passes by suppressing tokens and collects output logits.

        prompt: a list consisting of ImageInput and str. This is the input with respect to which the target tokens are to be explained.
        target: string target (ideally, the completion from the model)
        prompt_explain_indices: list of tokens to be suppressed (with or without similar tokens), useful for fewshot prompts
        max_batch_size: higher batch size -> more memory consumption but faster results
        configs: specify a custom list of chunks which are supposed to be suppressed together. Like conceptual suppression, but fully manual.
        save_configs_as: a json fileame into which configs can be saved for viewing layer. If none, nothing is saved.
        save_configs_only: if set to true, no forward passes would be performed and no logits would be returned. Set this to true when you just want to compute conceptual suppression configs.

        config example:
        ```
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
        ```
        """

        assert (prompt_explain_indices is not None and configs is not None) is False, 'Expected either one of prompt_explain_indices or configs to be None'

        if isinstance(prompt, str):
            prompt = [prompt]

        if isinstance(target, str):
            target = [target]

        prompt_embeddings = self.model.preprocess_inputs(prompt)
        prompt_length = prompt_embeddings.shape[1]
        target_embeddings = self.model.preprocess_inputs(target.copy())
        batch_size, prompt_seq_len, hidden_dim = prompt_embeddings.shape

        target_token_ids = self.model.preprocess_inputs(target, embed = False)[0][0].tolist()
        # prompt_explain_indices is the list of indices in the prompt and the target whose "importance" values are to be included in the explanation
        if prompt_explain_indices is None:
            prompt_explain_indices = [i for i in range(prompt_seq_len)]

        # total_seq_len is the total sequence length, including both the prompt and target
        total_seq_len = prompt_seq_len + len(target_token_ids)

        # target_token_indices is the list of indices which point to the target tokens
        ## notice how we're excluding the last token (-1), this is because we do not want the last target token in the input embeddings
        target_token_indices = list(
            range(0, prompt_length + len(target_token_ids) - 1)[
                prompt_length :
            ]
        )

        if configs is None:
            configs  = self.get_default_configs_for_forward_passes(
                prompt_explain_indices = prompt_explain_indices,
                target_token_indices = target_token_indices
            )
        else:
            assert configs[0] == {
                "suppression_token_index": [-1],
                "suppression_factor": [1.0],  # do not change
            }, f'Expected first element of configs to contain params which do no manipulation, but got: {configs[0]}'

        ## we are doing the same in the internal codebase, cropping out the last token
        input_embeddings = torch.cat(
            [
                prompt_embeddings,
                target_embeddings[:,:-1,:]  ## exclude last item in seq dim
            ],
            dim = 1
        )

        if self.conceptual_suppression_threshold is not None:
            self.conceptual_suppression = ConceptualSuppression(
                embeddings = input_embeddings,
                similarity_threshold=self.conceptual_suppression_threshold,
                modify_suppression_factor_based_on_cossim = self.modify_suppression_factor_based_on_cossim
            )

            configs = self.conceptual_suppression.modify_configs(
                configs = configs.copy()
            )

        if save_configs_as is not None:
            dict_to_json(dictionary=configs, filename = save_configs_as)
            print('saved configs:', save_configs_as)

        if save_configs_only is False:
            batches = list(chunks(configs, n = max_batch_size))

            """
            Notes on all_logits:
            - all_logits is a tensor of shape [len(configs), seq, embedding_dim]
            - first batch item of all_logits is the default output with no suppression
            - the 2nd batch item is the logit output after suppressing the first "chunk"

            Notes on Chunking:
            - a chunk is a group of tokens which are considered to be the part of a singular object in an images
            - this chunk logic translates to text too! we can suppress one sentence at a time (as a future option)
            """
            all_logits = self.collect_logits(
                batches = batches,
                input_embeddings = input_embeddings
            )

            ## we expect the batch size of the output logits to be the same as that of configs
            assert all_logits.shape[0] == len(configs)

            ## all_logits is supposed to have seq_len total_seq_len - 1 since the last token of the target is omitted
            assert all_logits.shape[1] == total_seq_len - 1, f'Expected total_seq_len - 1 to be the same as all_logits.shape[1] but got total_seq_len: {total_seq_len}, all_logits.shape[1]: {all_logits.shape[1]}'

            results = self.compile_results_from_configs_and_logits(
                logits = all_logits,
                configs = configs,
                target_token_ids=target_token_ids,
                target_token_indices= target_token_indices,
                prompt_length = prompt_length,
                prompt_explain_indices = prompt_explain_indices
            )

            return results
        else:
            return configs

    def compile_results_from_configs_and_logits(self, configs, logits, target_token_ids, target_token_indices, prompt_length, prompt_explain_indices):
        """
        1. We make a dict called results which stores the original logits i.e all_logits[0]
        2. By original logits we mean the output logits of the model when there was no attention manipulation
        3. We also add some other metadata which is gonna be useful to calculate the delta cross entropies (or any other method)
        4. suppressed_chunk_logits is a list that would contain the output logits for each "chunk" that was suppressed
        """
        results = {
            "original_logits": logits[0].cpu(),
            "target_token_ids": target_token_ids,
            "target_token_indices": target_token_indices,
            "prompt_length": prompt_length,
            "prompt_explain_indices": prompt_explain_indices,
            "suppressed_chunk_logits": []
        }

        ## starts at 1, because idx 0 contained the original logits
        idx = 1

        for config in configs[1:]:
            if 'original_token_index' in list(config.keys()):
                ## this means that conceptual suppression was applied
                data = {
                    "suppression_token_indices": config['original_token_index'],
                    "logits": logits[idx].cpu()
                }
            else:
                ## either conceptual suppression was not applied, or used custom chunk configs
                data = {
                    "suppression_token_indices": config['suppression_token_index'],
                    "logits": logits[idx].cpu()
                }
            results["suppressed_chunk_logits"].append(data)
            idx += 1

        return results


    def collect_logits(self, batches: list, input_embeddings):
        """
        1. we first initiate a list called all_logits which is supposed to contain all output logits from the model
        2. then we start a for loop which iterates over each "batch" config item:
            2.1 for each batch item, we first initiate an input embedding tensor of shape (batch_size, seq, embedding_dim)
            2.2 we extract the suppression_token_indices and suppression_factor values from the batch item and put them all into a single list
            2.3 we then run a forward pass with the params set
            2.4 we keep appending the logits into the list
        3. we concatenate all logits into one tensor along batch dim and then return it
        """
        all_logits = []
        for batch in batches:
            embeddings_batch = torch.cat(
                [input_embeddings for x in batch],
                dim = 0
            )

            suppression_token_indices = [
                x['suppression_token_index'] for x in batch
            ]

            suppression_factors = [
                x['suppression_factor'] for x in batch
            ]

            self.model.lm.suppression_factors = suppression_factors
            self.model.lm.suppression_token_indices = suppression_token_indices
            self.model.lm.manipulate_attn_scores_after_scaling = self.manipulate_attn_scores_after_scaling
            self.model.lm.layers = self.layers

            ## the actual forward pass with special attention manipulation params which have been already set
            logits = get_output_logits_from_embeddings(
                model = self.model, embeddings= embeddings_batch
            )

            all_logits.append(logits.cpu())

            ## set them back to original values
            self.model.lm.suppression_token_indices = None
            self.model.lm.suppression_factors = None

        return torch.cat(all_logits, dim = 0)

    def convert_prompt_and_target_to_token_ids(self, prompt: str, target: str):
        prompt_token_ids = self.tokenizer.encode(prompt)
        target_token_ids = self.tokenizer.encode(target)

        return {
            'prompt_token_ids': prompt_token_ids,
            'target_token_ids': target_token_ids
        }

    def get_default_configs_for_forward_passes(self, prompt_explain_indices, target_token_indices):
        batch_items = []
        batch_data = {
            "suppression_token_index": [-1],
            "suppression_factor": [1.0],  # do not change
        }

        batch_items.append(batch_data)
        '''
        Note that we exclude last target token in target_token_indices because nothing exists after it which is to be explained
        '''
        token_indices_to_suppress = (
            prompt_explain_indices + target_token_indices
        )

        for i in token_indices_to_suppress:

            batch_data = {
                "suppression_token_index": [i],
                "suppression_factor": [self.suppression_factor],
            }

            batch_items.append(batch_data)

        return batch_items
