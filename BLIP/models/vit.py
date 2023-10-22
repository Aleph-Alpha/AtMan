'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * Based on timm code base
 * https://github.com/rwightman/pytorch-image-models/tree/master/timm
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from timm.models.helpers import named_apply, adapt_input_conv
import tqdm
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def manipulate_attention_scores(self, attention_scores, modified_causal_attention_mask):
        # shift attention scores such that the min value is always 0
        # attention_scores.shape = batch_size, num_heads, seq, seq
        #attention_scores = attention_scores - attention_scores.min(-1).values.unsqueeze(
        #    3
        #)

        ## apply modified mask after repeating it along seq dim i.e attention_scores.shape[1]
        #TODO:
        #attention_scores = attention_scores * attention_mask_factors
        ## apply modified mask
        attention_scores = attention_scores * modified_causal_attention_mask

        #TODO:
        #attention_scores.masked_fill_(
        #    ~attention_mask.to(attention_scores.device), -10000.0
        #)

        return attention_scores

    def default_index_fn(self, relevance_matrix, target_token_index, num_target_token_ids):
        '''
        this is the indexing we use by default :)
        '''
        idx = -num_target_token_ids +  target_token_index
        # print(f'target_token_index: {target_token_index} num_target_token_ids: {num_target_token_ids} idx: {idx}')
        return relevance_matrix[idx,1:145] #proposed was [idx, :144] which is not suited for decoder!

    def forward(self, x, register_hook=False, suppression_matrix=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if suppression_matrix is not None:
            attn = self.manipulate_attention_scores(
                attention_scores=attn,
                modified_causal_attention_mask=suppression_matrix,
            )

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)


        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if True:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_grad_checkpointing=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

    def forward(self, x, register_hook=False, suppression_matrix=None):
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook, suppression_matrix=suppression_matrix))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 use_grad_checkpointing=False, ckpt_layer=0):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_grad_checkpointing=(use_grad_checkpointing and i>=depth-ckpt_layer)
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


    def get_similarity_matrix(self, a, b, eps=1e-8):
        """
        finds the cosine similarity matrix between each item of a w.r.t each item of b
        a and b are expected to be 2 dimensional (seq, hidden_dim)
        added eps for numerical stability
        source: https://stackoverflow.com/a/58144658
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def get_embedding_similarity_matrix(self, embeddings_batch):
        """
        Returns a similarity matrix of shape (batch_size, seq, seq) containing the cosine
        similarities of each token w.r.t every other token.
        """
        assert (
            embeddings_batch.ndim == 3
        ), f"Expected embeddings_batch to have 3 dimensions but got {embeddings_batch.ndim}"
        batch_size, seq_len = embeddings_batch.shape[0], embeddings_batch.shape[1]
        cossim_matrices = torch.zeros(
            embeddings_batch.shape[0],  # batch
            embeddings_batch.shape[1],  # seq
            embeddings_batch.shape[1],  # seq
        )

        with torch.no_grad():
            for batch_idx in range(batch_size):

                source_embeddings = embeddings_batch[batch_idx].float()
                sim_matrix = self.get_similarity_matrix(
                    a=source_embeddings, b=source_embeddings
                )
                cossim_matrices[batch_idx] = sim_matrix

        assert cossim_matrices.shape[0] == 1, f'Expected batch size to be 1 but got: {cossim_matrices.shape[0]}'
        assert cossim_matrices.shape[1] == cossim_matrices.shape[2], 'Expected a square matrix :('

        return cossim_matrices.clip(-1, 1)

    def get_suppression_factor_from_cosine_similarity(
        self, suppression_factor, cosine_similarity
    ):
        ## the formula we use for calculating the suppression factor for a conceptually similar token
        ## given a suppresion factor and the cossim of the similar token w.r.t the input token
        if 0 <= cosine_similarity.item() <= 1.0:
            x = (1 - suppression_factor) * (1 - cosine_similarity) + suppression_factor
        else:
            x = torch.tensor([1.0])
        return x

    def get_modified_causal_attention_mask(self, seq_len, suppression_token_indices, suppression_factors):
        ## attention_mask is basically the causal mask
        # attention_mask so far has shape [1, 1, seq, seq] in dtype bool
        # it is True whenever a value has to be masked out (i.e. not used in softmax)
        # convert such that 1 whenevser a value has to be kept and zero for masking out
        # there would be different attention masks for different batch items

        ## this fancy indexing on bias is hopefully equivalent to: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        #attention_mask = self.transformer.h[0].attn.attention.bias[:, :, 0 : seq_len, :seq_len]
        attention_mask_factors = torch.ones((1,1,seq_len+1,seq_len+1), dtype=torch.float32)


        for token_index, factor in zip(
            suppression_token_indices,
            suppression_factors,
        ):
            ## do nothing if token_index is -1 or None
            if token_index == -1 or token_index is None:
                continue

            attention_mask_factors[:, :, :, token_index+1] = torch.tensor(factor)

        return attention_mask_factors.to('cuda:0')

    def process_img(self, x,B,register_blk=-1, additional_indices=None, additional_suppression_factors=None):
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        #TODO: is this pos embed an issue?
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)


        suppression_matrix=None
        if additional_indices:
            suppression_matrix = self.get_modified_causal_attention_mask(900, additional_indices, additional_suppression_factors)


        for i,blk in enumerate(self.blocks):
            x = blk(x, register_blk==i,suppression_matrix=suppression_matrix)
        x = self.norm(x)
        return x

    def forward(self, x, register_blk=-1, suppression_factor=None, conceptual_suppression_threshold=None,
                limit_vision=9999, limit_suppressions=100, similarities=None):
        B = x.shape[0]

        orig_x = self.patch_embed(x)
        assert x.shape[0] == 1, "batch messes up suppressions"

        if suppression_factor:

            all_returns = [self.process_img(orig_x,B)]
            all_factors = []
            for i in tqdm.tqdm(range(min(limit_vision, 144))):
                suppression_token_index = i
                suppression_factor = suppression_factor
                similarity_scores = similarities[suppression_token_index].reshape(-1)
                assert (
                        similarity_scores.ndim == 1
                    ), f"Expected similarity_scores.ndim to be 1 but got: {similarity_scores.ndim}"

                additional_indices_bool = similarity_scores >= conceptual_suppression_threshold
                additional_indices = additional_indices_bool.nonzero()[0].tolist()
                #only top limit
                additional_indices = [l[1] for l in sorted(zip(similarity_scores[additional_indices],additional_indices),key=lambda x : -x[0])[:limit_suppressions]]
                #import pdb; pdb.set_trace()

                ## remove the index w.r.t which we calculated the scores (cossim 1) from the additional indices
                if suppression_token_index in additional_indices:
                    additional_indices.remove(suppression_token_index)

                ## -1 offset because first iter is skipped because item['suppression_token_index'] == [-1] is True
                additional_suppression_factors = [
                    self.get_suppression_factor_from_cosine_similarity(
                        suppression_factor = suppression_factor,
                        cosine_similarity = similarity_scores[additional_indices][i]
                    ).item()
                    for i in range(len(additional_indices))
                ]


                x = self.process_img(orig_x, B, additional_indices=additional_indices, additional_suppression_factors=additional_suppression_factors)

                all_returns.append(x)
                all_factors.append(additional_suppression_factors)

            # embedding_similarities = self.get_embedding_similarity_matrix(orig_x)
            # for i in tqdm.tqdm(range(min(limit_vision,embedding_similarities.shape[1]))):
            #     suppression_token_index = i
            #     suppression_factor = suppression_factor
            #     similarity_scores = embedding_similarities[0][suppression_token_index]
            #     assert (
            #             similarity_scores.ndim == 1
            #         ), f"Expected similarity_scores.ndim to be 1 but got: {similarity_scores.ndim}"

            #     additional_indices_bool = similarity_scores >= conceptual_suppression_threshold
            #     additional_indices = additional_indices_bool.nonzero().view(-1).tolist()
            #     #only top limit
            #     additional_indices = [l[1] for l in sorted(zip(similarity_scores[additional_indices],additional_indices),key=lambda x : -x[0])[:limit_suppressions]]
            #     #import pdb; pdb.set_trace()

            #     ## remove the index w.r.t which we calculated the scores (cossim 1) from the additional indices
            #     if suppression_token_index in additional_indices:
            #         additional_indices.remove(suppression_token_index)

            #     ## -1 offset because first iter is skipped because item['suppression_token_index'] == [-1] is True
            #     additional_suppression_factors = [
            #         self.get_suppression_factor_from_cosine_similarity(
            #             suppression_factor = suppression_factor,
            #             cosine_similarity = similarity_scores[additional_indices][i]
            #         ).item()
            #         for i in range(len(additional_indices))
            #     ]


            #     x = self.process_img(orig_x, B, additional_indices=additional_indices, additional_suppression_factors=additional_suppression_factors)

            #     all_returns.append(x)
            #     all_factors.append(additional_suppression_factors)
            return torch.cat(all_returns,dim=0),all_factors,[] #similarities.cpu().numpy().tolist()

        x = self.process_img(orig_x,B)


        return x

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
#     if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
#         model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
#         model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
#     if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
#         model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
#         model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))

        return new_pos_embed
    else:
        return pos_embed_checkpoint
