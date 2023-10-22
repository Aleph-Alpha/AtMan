from .magma import Magma
from .chefer.chefer_magma.magma import CheferMagma

import torch

class FakeMagma:
    def __init__(
        self,
        num_layers = 28,
        checkpoint_path = "./mp_rank_00_model_states.pt",
        mode = 'chefer',
        device = 'cuda:0'
    ):
        torch.cuda.empty_cache()

        self.modes = [
            'normal',
            'chefer',
            'captum'
        ]
        self.default_num_layers = 28

        assert mode in self.modes, f'Expected mode to be one of {self.modes} but got: {mode}'





        if num_layers == self.default_num_layers:
            fake_magma_num_layers = None
        else:
            fake_magma_num_layers = num_layers

        if mode == 'normal' or mode == 'captum':
            self.model = Magma.from_checkpoint(
                checkpoint_path = checkpoint_path,
                device = device,
                fake_magma_num_layers = fake_magma_num_layers
            ).eval()

        ## chefer
        else:
            self.model = CheferMagma.from_checkpoint(
                checkpoint_path = checkpoint_path,
                device = device,
                fake_magma_num_layers = fake_magma_num_layers
            ).eval()

        num_layers_in_model = len(self.model.lm.transformer.h)
        assert num_layers_in_model == num_layers, f'Expected magma to have {num_layers} layers :('



    # ## Adding some extra layers to an existing nn.ModuleList
    # def add_n_layers(self, n):
    #     for i in range(n):
    #         self.model.lm.transformer.h.append(
    #                 self.model.lm.transformer.h[-1] ## keep stacking the last layer
    #             )
    #     print(f'Added {n} layers successfully, with now a total of: {len(self.model.lm.transformer.h)} layers')

    ## prints model size etc
    def show_model_details(self):
        '''
        PROBLEM:
        self.model.parameters() is somehow not changing even after
        using self.add_n_layers
        '''

        total_params = sum(
            param.numel() for param in self.model.parameters()
        )
        a_billion = 1_000_000_000

        print(f'Model size: {total_params/a_billion} Billion params')
        print(f'Number of layers: {len(self.model.lm.transformer.h)}')
