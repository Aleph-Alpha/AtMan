import torch
import numpy as np
import cv2

from .chefer_magma.magma import CheferMagma

# create heatmap from mask on image
def show_cam_on_image(
    img,
    mask,
    mask_interpolation = cv2.INTER_NEAREST,
    cmap = cv2.COLORMAP_JET
):
    '''
    example usage:

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    for i in range(len(relevance_maps)):

        cam = show_cam_on_image(
            img = np.array(prompt[0].pil_image),
            mask = relevance_maps[i]['relevance_map'].reshape(12,12).numpy()
        )

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize = (10 , 4))
        fig.suptitle('Target token:'+ cm.magma.tokenizer.decode([relevance_maps[i]['target_token_id']]))
        ax[0].imshow(prompt[0].pil_image)
        ax[1].imshow(cam)
        ax[2].imshow(relevance_maps[i]['relevance_map'].reshape(12,12).numpy())
        plt.show()
    ```
    '''
    if mask.shape == (12,12):
        mask = cv2.resize(
            mask,
            dsize = (img.shape[1], img.shape[0]),
            interpolation = mask_interpolation
        )

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cmap)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)/255
    cam = cam / np.max(cam)
    return cam

class CheferMethod:
    '''
    Example usage:
    ```python
    cm = CheferMethod(
        magma = model,
        device = 'cuda:0'
    )

    prompt = [
        ## supports urls and path/to/image
        ImageInput('./samples/el2.png'),
        'A picture of an'
    ]
    embeddings = model.preprocess_inputs(prompt.copy())
    target = ' Elephant and a zebra'

    relevance_maps = cm.run(
        embeddings = embeddings,
        target = target
    )
    ```
    '''
    def __init__(self, magma: CheferMagma, device: str):
        ## CheferMagma is a modified version of magma which contains backward hooks on attention
        self.magma = magma
        self.device = device
        self.magma = self.magma.to(self.device).eval()

    # rule 5 from paper
    def avg_heads(self, cam, grad):
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = grad * cam

        ## they are clamping values between zero and +ve values (kinda like a relu)
        cam = cam.clamp(min=0).mean(dim=0)
        return cam

    # rule 6 from paper
    def apply_self_attention_rules(self, R_ss, cam_ss):
        R_ss_addition = torch.matmul(cam_ss, R_ss)
        return R_ss_addition

    def default_index_fn(self, relevance_matrix, target_token_index, num_target_token_ids):
        '''
        this is the indexing we use by default :)
        '''
        idx = -num_target_token_ids +  target_token_index
        # print(f'target_token_index: {target_token_index} num_target_token_ids: {num_target_token_ids} idx: {idx}')
        return relevance_matrix[idx, :144]

    def run(self, embeddings, target: str, custom_index_fn = None):
        '''
        Steps:
        0. make sure input embeddings batch size is 1
        1. forward pass through model (with grads)
        2. convert target string to token ids using tokenizer
        3. calculate relevance for each target token
        4. return nice list where each item is a dictionary containing:
            - target token index (int)
            - target token id (int)
            - relevance map (numpy array)
        '''

        ## step 0 safety first
        assert embeddings.shape[0] ==1 , f'Expected batch size to be 1 but got: {embeddings.shape[0]}'

        ## find num tokens in prompt
        num_tokens_in_prompt = embeddings.shape[1]

        ## target embeddings
        target_embeddings = self.magma.preprocess_inputs(
            [target]
        ).to(self.device)

        ## combine prompt and target embeddings along seq dim
        combined_embeddings = torch.cat(
            [
                embeddings.to(self.device),
                ## exclude last item in seq dim
                target_embeddings[:,:-1,:]
            ],
            dim = 1
        )

        ## step 1
        ## output_logits.shape: (batch, seq, vocabulary)
        output_logits = self.magma.forward(
            input_embeddings = combined_embeddings,
        ).logits

        ## step 2
        target_token_ids = self.magma.tokenizer.encode(target)
        num_target_token_ids = len(target_token_ids)

        ## completion_logits.shape: (num_target_token_ids, vocabulary)
        ## -1 because next token
        completion_logits = output_logits[0,num_tokens_in_prompt-1:, :]
        assert completion_logits.shape[0] == num_target_token_ids, f'Expected completion_logits.shape[0] to be: {num_target_token_ids} but got shape: {completion_logits.shape}'

        relevance_maps = []

        for target_token_index in range(num_target_token_ids):

            ## make a onehot vector of shape (batch_size, vocabulary)
            ## set the index of the target token id as 1
            one_hot = np.zeros((1, output_logits.shape[-1]), dtype=np.float32)
            one_hot[0, target_token_ids[target_token_index]] = 1.

            ## convert it to a torch tensor
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)

            ## dot product (?)
            one_hot = torch.sum(one_hot.to(self.device) * completion_logits[target_token_index, :].unsqueeze(0))

            ## shortcut one-hot, possibly doing the same thing, but will NOT use it for now
            # one_hot = completion_logits[target_token_index, target_token_ids[target_token_index]]

            ## make sure there are no older grads which might accumulate
            self.magma.zero_grad()
            one_hot.backward(retain_graph=True)

            num_tokens = self.magma.lm.transformer.h[0].attn.get_attention_map().shape[-1]

            R = torch.eye(num_tokens, num_tokens).to(self.device)

            for blk in self.magma.lm.transformer.h:

                grad = blk.attn.get_attn_gradients().detach()
                cam = blk.attn.get_attention_map().detach()
                cam = self.avg_heads(cam, grad)
                R += self.apply_self_attention_rules(R.to(self.device).float(), cam.to(self.device).float())

            ## apply custom indexing on relevance "matrix"(?)
            ## dont know which is the correct way to index this for decoder models
            if custom_index_fn is not None:
                relevance_map = custom_index_fn(
                    relevance_matrix = R,
                    target_token_index = target_token_index,
                    num_target_token_ids = num_target_token_ids
                ).cpu().detach()

            else:
                relevance_map = self.default_index_fn(
                    relevance_matrix = R,
                    target_token_index= target_token_index,
                    num_target_token_ids = num_target_token_ids
                ).cpu().detach()

            relevance_maps.append(relevance_map)

        results = []
        for i in range(len(relevance_maps)):
            data = {
                'target_token_index': i,
                'target_token_id': target_token_ids[i],
                'relevance_map': relevance_maps[i]
            }
            results.append(data)

        return results
