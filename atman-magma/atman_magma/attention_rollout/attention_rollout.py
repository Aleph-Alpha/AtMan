import numpy as np
import torch

from magma.image_input import ImageInput

class AttentionRolloutMagma:
    def __init__(self, model):
        """
        usage:
        ```
        import seaborn as sns
        from atman_magma.magma  import Magma
        from atman_magma.attention_rollout.attention_rollout import AttentionRolloutMagma


        print('loading model...')
        device = 'cuda:0'
        model = Magma.from_checkpoint(
            checkpoint_path = "mp_rank_00_model_states.pt",
            device = device
        )
        prompt = ['hello world my name is Ben and i love burgers']
        
        attention_rollout = AttentionRolloutMagma(model = model)
        embeddings = attention_rollout.model.preprocess_inputs(prompt) ## prompt is a list

        y = attention_rollout.run(embeddings)
        sns.heatmap(y)
        ```
        """
        self.model = model.eval()
        self.next_token_index = -1
        
    def forward_pass(self, embeddings):
        with torch.no_grad():
            return self.model.lm(inputs_embeds  = embeddings, output_attentions = True)
        
    def run(self, embeddings, return_next_token_output_only: bool = False):
        assert embeddings.ndim == 3, 'Expected tensor with 3 dims: (Batch, Sequence, Embedding)'
        outputs = self.forward_pass(embeddings = embeddings)
        
        return self.get_attention_rollout_from_model_outputs(outputs, return_next_token_output_only  = return_next_token_output_only)
    
    def get_attention_rollout_from_model_outputs(self, outputs, return_next_token_output_only: bool = True):
        all_attentions = outputs.attentions
        _attentions = [att.cpu().detach().numpy() for att in all_attentions]
        attentions_mat = np.asarray(_attentions)[:,0]

        # print(attentions_mat.shape)

        res_att_mat = attentions_mat.sum(axis=1)/attentions_mat.shape[1]
        res_att_mat = res_att_mat + np.eye(res_att_mat.shape[1])[None,...]
        res_att_mat = res_att_mat / res_att_mat.sum(axis=-1)[...,None]
        # print('post res shape',res_att_mat.shape)

        joint_attentions = np.zeros(res_att_mat.shape)

        layers = joint_attentions.shape[0]
        joint_attentions[0] = res_att_mat[0]
        for i in np.arange(1,layers):
            joint_attentions[i] = res_att_mat[i].dot(joint_attentions[i-1])
            
        if return_next_token_output_only:
            return joint_attentions[:, self.next_token_index,:]
        else:
            return joint_attentions
    
    def run_on_image(self, prompt, target, manipulate_last_n_tokens: int = None):
        assert isinstance(prompt[0], ImageInput), f'Expected the first prompt item to be an ImageInput but got: {type(prompt[0])}'
        assert isinstance(prompt[1], str), f'Expected the second prompt item to be an str but got: {type(prompt[1])}'
        
        prompt_embeddings = self.model.preprocess_inputs(prompt) ## prompt is a list
        target_embeddings = self.model.preprocess_inputs(target)
        # print('len target:', target_embeddings.shape[1])
        embeddings = torch.cat(
            [
                prompt_embeddings,
                target_embeddings[:,:-1,:] ## exclude last token
            ],
            dim =1
        )
        
        joint_attentions = self.run(embeddings, return_next_token_output_only = False)
        
        target_token_indices = [i for i in range(prompt_embeddings.shape[1]-1, embeddings.shape[1])]
        
        heatmap = np.zeros((12,12))
        
        for idx in target_token_indices:
            heatmap += joint_attentions[0,idx,1:145].reshape(12,12)
           
        
        if manipulate_last_n_tokens is not None:
            heatmap[-1,-manipulate_last_tokens:]=0.  ## set explanation values of last 2 tokens to 0
        return heatmap
        