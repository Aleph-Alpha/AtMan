import torch
from .outputs import DeltaCrossEntropiesOutput, DeltaLogitsOutput

def get_delta_cross_entropies(output: dict, square_outputs = False, custom_chunks = False, temperature: float = None):
    
    ## safety first :>
    assert 'original_logits' in list(output.keys()), 'Expected key: original_logits to be present in the output dict'
    assert 'suppressed_chunk_logits' in list(output.keys()), 'Expected key: suppressed_chunk_logits to be present in the output dict'
    
    ## I know this is redundant code, but I want to make it readable

    ## the original output logits from the model without any attention manipulation
    original_logits = output['original_logits']

    ## labels for CE loss are the indices of the logits with respect to which
    ## we have to calculate the cross entropy losses
    labels_for_ce_loss = torch.tensor(output['target_token_ids'])
    target_token_ids = output['target_token_ids']
    target_token_indices = output['target_token_indices']
    prompt_explain_indices = output['prompt_explain_indices']
    prompt_length =output['prompt_length']
    
    assert original_logits.ndim == 2, f'Expected original_logits to have 2 dims: (seq, vocab_size) but got shape: {original_logits.shape}'
    
    ## this is where the bug was ╥_╥
    target_token_indices_in_logits = [
        i for i in range(prompt_length - 1, len(original_logits))
    ]
    original_target_token_logits = original_logits[target_token_indices_in_logits, :]

    ## default loss is of shape len(target_token_indices)

    if temperature is not None:
        original_target_token_logits /= temperature

    default_loss = torch.nn.functional.cross_entropy(
            original_target_token_logits,
            labels_for_ce_loss.to(original_target_token_logits.device),
            reduction="none",
    ).cpu()

    all_chunk_indices = [
        x['suppression_token_indices'] for x in output['suppressed_chunk_logits']
    ]

    ## now lets compute results!
    result = []

    delta_cross_entropies_matrix = torch.zeros(
        (
            # (len(output['prompt_explain_indices']) + len(target_token_ids) - 1),
            len(output['suppressed_chunk_logits']),
            len(target_token_ids)
        ),
        dtype=torch.float,
    )

    for i in range(len(output['suppressed_chunk_logits'])):
        result  = output['suppressed_chunk_logits'][i]
        target_logits = result['logits'][target_token_indices_in_logits, :]

        if temperature is not None:
            target_logits /= temperature
        
        loss = torch.nn.functional.cross_entropy(
            target_logits,
            labels_for_ce_loss.to(original_target_token_logits.device),
            reduction="none",
        ).cpu()                                 
        if square_outputs is True:
            delta_cross_entropies_matrix[i] = (default_loss - loss)**2
        else:
            delta_cross_entropies_matrix[i] = (default_loss - loss)

    result = []

    ## for each target token
    for i in range(len(target_token_ids)):
        data = {
            "target_token_id": target_token_ids[i],
            "target_token_index": i,
            "explanation": [],
        }

        ## for each token preceding the target token
        count = 0

        if custom_chunks == True:
            for  chunk_indices in all_chunk_indices:
                data['explanation'].append(
                    {
                        "token_index": chunk_indices,
                        "value":delta_cross_entropies_matrix[count, i].item()
                    }
                )
                count +=1
        else:
            """
            This is the original schema
            """
            for j in (prompt_explain_indices + target_token_indices[:i]):
                data['explanation'].append(
                    {
                        "token_index": j,
                        "value":delta_cross_entropies_matrix[count, i].item()
                    }
                )
                count +=1


        result.append(data)

    return DeltaCrossEntropiesOutput(
        data = result
    )


#########################################################
#################### LOGIT LOSS #########################
def get_delta_logits(output: dict, square_outputs = False):
    
    ## safety first :>
    assert 'original_logits' in list(output.keys()), 'Expected key: original_logits to be present in the output dict'
    assert 'suppressed_chunk_logits' in list(output.keys()), 'Expected key: suppressed_chunk_logits to be present in the output dict'
    
    ## I know this is redundant code, but I want to make it readable

    ## the original output logits from the model without any attention manipulation
    original_logits = output['original_logits']

    ## labels for CE loss are the indices of the logits with respect to which
    ## we have to calculate the cross entropy losses
    labels_for_ce_loss = torch.tensor(output['target_token_ids'])
    target_token_ids = output['target_token_ids']
    target_token_indices = output['target_token_indices']
    prompt_explain_indices = output['prompt_explain_indices']
    prompt_length =output['prompt_length']
    
    assert original_logits.ndim == 2, f'Expected original_logits to have 2 dims: (seq, vocab_size) but got shape: {original_logits.shape}'
    
    ## this is where the bug was ╥_╥
    target_token_indices_in_logits = [
        i for i in range(prompt_length - 1, len(original_logits))
    ]
    original_target_token_logits = original_logits[target_token_indices_in_logits, :]

    ## default loss is of shape len(target_token_indices)

    original_logit_values = []

    for seq_item, label_idx in zip(original_target_token_logits, labels_for_ce_loss):
        original_logit_values.append(
            seq_item[label_idx.item()].item()
        )
    
    original_logit_values = torch.tensor(original_logit_values)


    all_chunk_indices = [
        x['suppression_token_indices'] for x in output['suppressed_chunk_logits']
    ]

    ## now lets compute results!
    result = []

    delta_logits_matrix = torch.zeros(
        (
            # (len(output['prompt_explain_indices']) + len(target_token_ids) - 1),
            len(output['suppressed_chunk_logits']),
            len(target_token_ids)
        ),
        dtype=torch.float,
    )


    for i in range(len(output['suppressed_chunk_logits'])):
        result  = output['suppressed_chunk_logits'][i]
        target_logits = result['logits'][target_token_indices_in_logits, :]

        logit_values = []
        for seq_item, label_idx in zip(target_logits, labels_for_ce_loss):
            logit_values.append(
                seq_item[label_idx.item()].item()
            )
        
        logit_values = torch.tensor(logit_values)
        if square_outputs is True:
            delta_logits_matrix[i] = (original_logit_values - logit_values)**2
        else:
            delta_logits_matrix[i] = (original_logit_values - logit_values)

    result = []

    ## for each target token
    for i in range(len(target_token_ids)):
        data = {
            "target_token_id": target_token_ids[i],
            "target_token_index": i,
            "explanation": [],
        }

        ## for each token preceding the target token
        count = 0

        for  chunk_indices in all_chunk_indices:
            data['explanation'].append(
                {
                    "chunk_indices": chunk_indices,
                    "value":delta_logits_matrix[count, i].item()
                }
            )
            count +=1

        result.append(data)

    return DeltaLogitsOutput(
        data = result
    )