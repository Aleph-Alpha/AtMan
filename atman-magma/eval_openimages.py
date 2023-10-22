from atman_magma.magma  import Magma
from atman_magma.explainer import Explainer
from atman_magma.logit_parsing import get_delta_cross_entropies
from atman_magma.openimages_eval import run_eval


from multimodal_explain_eval.dataloader import DataLoader
from multimodal_explain_eval.utils import load_json_as_dict

import yaml

with open("config.yml", "r") as stream:
    config = yaml.safe_load(stream)

conc_sup_values = config["hyperparams"]["conc_sup_values"]
suppression_factor_values = config["hyperparams"]["suppression_factor_values"]


metadata = load_json_as_dict(
    filename = config['files']['metadata_filename']
)

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
    len(conc_sup_values) * len(suppression_factor_values) * num_images
)
print(f"Will run a total of: {num_total_explanations} explanations")

print('output folder names:')
folder_names = []
for x in conc_sup_values:
    for y in suppression_factor_values:
        folder_name = f"{config['files']['output_dir']}/con_sup_thres_{round(x, 2) if x is not None else None}_suppression_factor_{round(y, 2) if y is not None else None}"
        print(folder_name)
        folder_names.append(
            folder_name
        )

print('loading model...')
model = Magma.from_checkpoint(
    checkpoint_path = config['model']['checkpoint_path'],
    device = config['model']['device']
)

folder_idx = 0

for conceptual_suppression_threshold in conc_sup_values:
    for suppression_factor in suppression_factor_values:

        ex = Explainer(
            model = model, 
            device = config['model']['device'], 
            tokenizer = model.tokenizer, 
            conceptual_suppression_threshold = conceptual_suppression_threshold,
            suppression_factor = suppression_factor
        )

        run_eval(
            explainer = ex,
            metadata = metadata,
            dataloader = dataloader,
            logit_parsing_fn=get_delta_cross_entropies,
            output_folder = folder_names[folder_idx],
            max_batch_size = config['model']['max_batch_size'],
            text_prompt = config['hyperparams']['prompt_text'],
            use_lowercase_target=True,
            auto_decide_a_or_an=True,
            progress=True,
            square_outputs = False,
            num_total_explanations=num_total_explanations,
            prompt_explain_indices = [i for i in range(144)],
            save_configs_only=config['hyperparams']['save_configs_only'],
            save_configs_dir='./configs'
        )
        folder_idx += 1

print('eval complete :)')