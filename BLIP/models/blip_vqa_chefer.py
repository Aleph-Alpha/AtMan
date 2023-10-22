from models.med import BertConfig, BertModel, BertLMHeadModel
from models.blip import create_vit, init_tokenizer, load_checkpoint

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np

class BLIP_VQA(nn.Module):
    def __init__(self,
                 med_config = 'configs/med_config.json',
                 image_size = 480,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1)
        self.tokenizer = init_tokenizer()

        encoder_config = BertConfig.from_json_file(med_config)
        print(f"BERTY CONFIG {encoder_config}")
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)

        decoder_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config)

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


    def forward_chefer(self, image, question, answer):

        self.zero_grad()
        self.visual_encoder.zero_grad()
        self.text_encoder.zero_grad()
        self.text_decoder.zero_grad()
        image_embeds = self.visual_encoder(image, suppression_factor=None,
                                                                  conceptual_suppression_threshold=None,
                                                                  limit_vision=None, limit_suppressions=None, similarities=None
                                                                  )

        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        question = self.tokenizer(question, padding='longest', truncation=True, max_length=35,
                                  return_tensors="pt").to(image.device)
        question.input_ids[:,0] = self.tokenizer.enc_token_id

        num_repeat=image_embeds.shape[0]

        question.attention_mask = question.attention_mask.repeat(num_repeat,1)
        question.input_ids = question.input_ids.repeat(num_repeat,1)

        relevance_maps = []


        answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(image.device)
        answer.input_ids[:,0] = self.tokenizer.bos_token_id
        answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)

        question_output = self.text_encoder(question.input_ids,
                                            attention_mask = question.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,
                                            return_dict = True,output_attentions=True)

        n = [1]
        question_states = []
        question_atts = []
        for b, n in enumerate(n):
            question_states += [question_output.last_hidden_state[b]]*n
            question_atts += [question.attention_mask[b]]*n
        question_states = torch.stack(question_states,0)
        question_atts = torch.stack(question_atts,0)

        answer_output = self.text_decoder(answer.input_ids,
                                            attention_mask = answer.attention_mask,
                                            encoder_hidden_states = question_states,
                                            encoder_attention_mask = question_atts,
                                            labels = answer_targets,
                                            return_dict = True,
                                            output_attentions=True,
                                            reduction = 'none',
                                            )

        answer_output.loss.backward(retain_graph=True)
        #self.visual_encoder.blocks[0].attn.attn_gradients
        #self.visual_encoder.blocks[0].attn.attention_map

        #self.text_encoder.encoder.layer[0].crossattention.self.attention_map.shape
        #self.text_encoder.encoder.layer[0].crossattention.self.attn_gradients.shape

        #import pdb; pdb.set_trace()

        R = torch.eye(901, 901).cuda()

        for blk in self.visual_encoder.blocks:

            grad = blk.attn.attn_gradients.detach()
            cam = blk.attn.attention_map.detach()
            cam = self.avg_heads(cam, grad)
            R += self.apply_self_attention_rules(R.cuda().float(), cam.cuda().float())

        return R.to('cpu')[0][1:].reshape(30,30)






    def forward(self, image, question, suppression_factor=None, conceptual_suppression_threshold=None, answer=None, n=None, weights=None, train=True,
                inference='rank', k_test=128, target_string="woman", limit_vision = 902, limit_suppressions=100, similarities=None):

        image_embeds, all_factors, embedsim = self.visual_encoder(image, suppression_factor=suppression_factor,
                                                                  conceptual_suppression_threshold=conceptual_suppression_threshold,
                                                                  limit_vision=limit_vision, limit_suppressions=limit_suppressions, similarities=similarities)

        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)


        question = self.tokenizer(question, padding='longest', truncation=True, max_length=35,
                                  return_tensors="pt").to(image.device)
        question.input_ids[:,0] = self.tokenizer.enc_token_id


        num_repeat=image_embeds.shape[0]

        question.attention_mask = question.attention_mask.repeat(num_repeat,1)
        question.input_ids = question.input_ids.repeat(num_repeat,1)

        question_output = self.text_encoder(question.input_ids,
                                            attention_mask = question.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,
                                            return_dict = True)

        num_beams = 1

        target_ids = self.tokenizer.encode(target_string)[:-1][:4] #max 3 target-tokens


        question_states = question_output.last_hidden_state.repeat_interleave(num_beams,dim=0)
        question_atts = torch.ones(question_states.size()[:-1],dtype=torch.long).to(question_states.device)
        model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask":question_atts}


        bos_ids = torch.full((image.size(0),1),fill_value=self.tokenizer.bos_token_id,device=image.device)
        bos_ids = bos_ids.repeat(num_repeat,1)

        # outputs = self.text_decoder.generate(input_ids=bos_ids,
        #                                      max_length=len(target_ids),
        #                                      min_length=1,
        #                                      num_beams=1,
        #                                      eos_token_id=self.tokenizer.sep_token_id,
        #                                      pad_token_id=self.tokenizer.pad_token_id,
        #                                      **model_kwargs)

        #import pdb; pdb.set_trace()

        output_mask = torch.ones(len(target_ids))
        #output_mask[0] = 0
        output_mask = output_mask.repeat(num_repeat,1).to('cuda')
        target_ids = torch.tensor(target_ids).repeat(num_repeat,1).to('cuda')


        #import pdb ; pdb.set_trace()
        # answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(image.device)
        # answer.input_ids[:,0] = self.tokenizer.bos_token_id
        # answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)
        target_ids[:,0] = self.tokenizer.bos_token_id
        #answer_targets = target_ids.masked_fill(target_ids == self.tokenizer.pad_token_id, -100)
        #import pdb; pdb.set_trace()
        print(f"target {target_string} {self.tokenizer.decode(target_ids[0,1:])}")
        #import pdb; pdb.set_trace()

        return self.text_decoder(target_ids,
                          #attention_mask = output_mask,
                          labels = target_ids,  return_dict = True,  reduction = 'none',  **model_kwargs), all_factors, embedsim

        import pdb;pdb.set_trace()
        output = self.text_decoder(bos_ids,
                                attention_mask = input_atts,
                                labels = targets_ids,
                                return_dict = True,
                                reduction = 'none',
                                **model_kwargs)


        # print(f'bos inputs: {bos_ids.shape} {question_states.shape}')
        # outputs = self.text_decoder.generate(input_ids=bos_ids,
        #                                      max_length=10,
        #                                      min_length=1,
        #                                      num_beams=num_beams,
        #                                      eos_token_id=self.tokenizer.sep_token_id,
        #                                      pad_token_id=self.tokenizer.pad_token_id,
        #                                      **model_kwargs)

        answers = []
        for output in outputs:
            answer = self.tokenizer.decode(output, skip_special_tokens=True)
            answers.append(answer)
        return answers




    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):

        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token

        start_output = self.text_decoder(start_ids,
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,
                                         return_dict = True,
                                         reduction = 'none')
        logits = start_output.logits[:,0,:] # first token's logit

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)
        input_atts = torch.cat(input_atts,dim=0)

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.text_decoder(input_ids,
                                   attention_mask = input_atts,
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,
                                   labels = targets_ids,
                                   return_dict = True,
                                   reduction = 'none')

        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques,k)

        max_topk_ids = log_probs_sum.argmax(dim=1)
        max_ids = topk_ids[max_topk_ids>=0,max_topk_ids]

        return max_ids


def blip_vqa(pretrained='',**kwargs):
    model = BLIP_VQA(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
#         assert(len(msg.missing_keys)==0)
    return model


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))
