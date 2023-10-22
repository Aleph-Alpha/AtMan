from datetime import datetime
from atman_magma.magma  import Magma
from atman_magma.explainer import Explainer
from atman_magma.logit_parsing import get_delta_cross_entropies
from atman_magma.openimages_eval import run_eval


from multimodal_explain_eval.dataloader import DataLoader
from multimodal_explain_eval.utils import load_json_as_dict

import  sys

from determined_eval_wrapper import process_determined_hparams

import os
import determined as det


startTime = datetime.now()


info = det.get_cluster_info()
if info is None:
    hparams = {}
else:
    hparams = info.trial.hparams if info.task_type == "TRIAL" else {}
params = process_determined_hparams(hparams)



if __name__ == "__main__":
    conc_sup_values = str(params.conc_sup).split(',')
    suppression_factor_values = str(params.sup_fact).split(',')

    metadata = load_json_as_dict('metadata.json')

    keys_to_delete = []
    for x in metadata.keys():
        if metadata[x]['count'] == 0:
            keys_to_delete.append(x)

    for key in keys_to_delete:
        del metadata[key]


    dataloader = DataLoader(
        metadata=metadata
    )

    num_images = 0
    for key in metadata:
        num_images += metadata[key]["count"]
        # print(f"{key:<17}:", metadata[key]["count"])

    print(f"total {num_images} pairs across {len(metadata)} classes")

    num_total_explanations = (
        len(conc_sup_values) *len(suppression_factor_values) * num_images
    )
    print(f"Will run a total of: {num_total_explanations} explanations")

    print('output folder names:')

    all_layers = [None]
    if params.layers is not None and params.layers != "None":
        all_layers = params.layers
    # else:

    for layers in all_layers:
        for x in conc_sup_values:
            for y in suppression_factor_values:
                print(f"CONFIG {x} {y} // {len(conc_sup_values)} {len(suppression_factor_values)}")
                if x == "None":
                    conc_sup_value = None
                else:
                    conc_sup_value = float(x)
                suppression_factor_value = float(y)
                folder_name = f"{params.result_path}/con_sup_thres_{str(x) if x is not None else None}_suppression_factor_{str(y) if y is not None else None}_layer_{','.join([str(l) for l in layers]) if layers is not None else 'None'}"

                print('loading model...')
                model = Magma.from_checkpoint(
                    checkpoint_path = "./mp_rank_00_model_states.pt",
                    device = 'cuda'
                )
                model = model.eval()

                folder_idx = 0

                num_total_explanations = num_images
                ex = Explainer(
                    model = model,
                    device = 'cuda',
                    tokenizer = model.tokenizer,
                    conceptual_suppression_threshold = conc_sup_value,
                    suppression_factor = suppression_factor_value,
                    layers=layers
                )

                run_eval(
                    explainer = ex,
                    metadata = metadata,
                    dataloader = dataloader,
                    logit_parsing_fn=get_delta_cross_entropies,
                    output_folder = folder_name,
                    max_batch_size = 144,
                    text_prompt =  'This is a picture of ',
                    use_lowercase_target=True,
                    auto_decide_a_or_an=True,
                    progress=True,
                    square_outputs = False,
                    num_total_explanations=num_total_explanations,
                    prompt_explain_indices = [i for i in range(144)]
                )
                folder_idx += 1

                print('eval complete :)')



    print("ALL DONE")
