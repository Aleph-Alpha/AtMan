from atman_magma.explainer import Explainer
from atman_magma.logit_parsing import get_delta_cross_entropies
from atman_magma.outputs import DeltaCrossEntropiesOutput, DeltaLogitsOutput
import numpy as np

from typing import  Union
import matplotlib.pyplot as plt
from .sentence_wise_suppression import SentenceWiseSuppression
from .utils import dict_to_json

class DocumentQAExplanationOutput:
    def __init__(self, postprocessed_output):
        self.data = postprocessed_output
    
    def save_as(self, filename: str):
        dict_to_json(dictionary=self.data, filename=filename)

    def __repr__(self):
        return str(self.data)

class DocumentQAExplainer:
    def __init__(
        self,
        model,
        document: str, 
        explanation_delimiters: str = '\n',
        device = 'cuda:0',
        suppression_factor = 0.0,
    ):
        
        self.explainer = Explainer(
            model = model, 
            device = device, 
            tokenizer = model.tokenizer, 
            conceptual_suppression_threshold = None,
            suppression_factor = suppression_factor
        )
        
        self.document = document

        self.instruction_before_document = 'Read the contents of the document carefully and answer the questions below.\n'
        
        self.prompt_prefix = f'''{self.instruction_before_document}{self.document}'''
        
        self.sentence_wise_suppression = SentenceWiseSuppression(
            prompt = self.prompt_prefix,
            tokenizer_encode_fn=model.tokenizer.encode,
            tokenizer_decode_fn= model.tokenizer.decode,
            delimiters= explanation_delimiters ## for now one token only
        )

        self.custom_chunk_config = self.sentence_wise_suppression.get_config(self.explainer.suppression_factor)
        self.instruction_after_document = 'Now answer the question below based on the document given above.'

    def get_prompt(self, question):
        prompt = [
            f'''{self.prompt_prefix}
Question: {question}
Answer:'''
        ]
        return prompt

    def run(
        self,
        question: str, 
        expected_answer: str,
        max_batch_size: int = 25
    ):
        prompt = self.get_prompt(question=question)
                
        logit_outputs = self.explainer.collect_logits_by_manipulating_attention(
            prompt = prompt.copy(),
            target = expected_answer,
            max_batch_size=max_batch_size,
            configs = self.custom_chunk_config
        )
        
        output = get_delta_cross_entropies(
            output = logit_outputs,
            square_outputs = True,
            custom_chunks = True
        )
        
        return output

    def postprocess(self, output: Union[DeltaLogitsOutput, DeltaCrossEntropiesOutput]):

        split_prompt =  self.sentence_wise_suppression.get_split_prompt()
        chunks = None
        
        all_values = []
        for target_token_idx in range(len(output.data)):
            single_explanation = output.data[target_token_idx]

            values = np.array([x['value'] for x in single_explanation['explanation']])
            all_values.append(values)
            if chunks is None:
                # chunks = [
                #     f'Chunk {i}: {split_prompt[i]}'.replace('.', '.\n') for i in range(len(split_prompt))
                # ]
                chunks = [
                    f'{split_prompt[i]}'.replace('.', '.\n') for i in range(len(split_prompt))
                ]

        data = []

        for chunk, value in zip(chunks, sum(all_values)):
            x = {
                "chunk":chunk,
                "value":value
            }
            data.append(x)

        return DocumentQAExplanationOutput(postprocessed_output=data)

    def get_chunks_and_values(self, postprocessed_output):
        chunks = [x['chunk'] for x in postprocessed_output.data]
        values = [x['value'] for x in postprocessed_output.data]

        return chunks, values

    def show_output(self, output, question, expected_answer, save_as: str = None, figsize = (15, 11), fontsize= 20):
        postprocessed_output = self.postprocess(output)
        chunks , values = self.get_chunks_and_values(
            postprocessed_output=postprocessed_output
        )

        fig = plt.figure(figsize = figsize)
        plt.barh(chunks[::-1], values[::-1]) ## reversed to read from top to bottom
        plt.yticks(rotation = 0)
        plt.xticks(fontsize = fontsize)
        plt.grid()
        plt.title(f'Question: {question}\nAnswer: {expected_answer}', fontsize = fontsize)
        plt.show()

        if save_as is not None:
            plt.tight_layout()
            fig.savefig(save_as)
            print('saved:', save_as)
