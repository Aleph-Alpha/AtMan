from multimodal_explain_eval.eval import run_eval
from multimodal_explain_eval.dataloader import DataLoader
from multimodal_explain_eval.utils import load_json_as_dict

import os
import yaml
import logging
from generator import Generator

with open("config.yml", "r") as stream:
    config = yaml.safe_load(stream)

logger = logging.getLogger()
handler = logging.FileHandler(config['logfile'])
logger.addHandler(handler)
logger = logging.getLogger()
logger.info(f"Running with config: {config}")


conc_sup_values = config["hyperparams"]["conc_sup_values"]
suppression_factor_values = config["hyperparams"]["suppression_factor_values"]
RESULT_DIR = config["files"]["result_dir"]
JSON_DIR = config["files"]["raw_jsons_folder"]
metadata = load_json_as_dict(config["files"]["metadata_filename"])

num_images = 0

for key in metadata:
    num_images += metadata[key]["count"]
    # print(f"{key:<17}:", metadata[key]["count"])

print(f"total {num_images} pairs across {len(metadata)} classes")

num_total_explanations = (
    len(conc_sup_values) * len(suppression_factor_values) * num_images
)
print(f"Will run a total of: {num_total_explanations} explanations")

folder_names = []
json_folder_names = []
assert os.path.exists(RESULT_DIR) == True, f"Expected folder to exist:{RESULT_DIR}"

for x in conc_sup_values:
    for y in suppression_factor_values:
        folder_names.append(
            f"{RESULT_DIR}/con_sup_thres_{round(x, 2) if x is not None else None}_suppression_factor_{round(y, 2) if x is not None else None}"
        )
        json_folder_names.append(
            f"{JSON_DIR}/con_sup_thres_{round(x, 2) if x is not None else None}_suppression_factor_{round(y, 2) if x is not None else None}"
        )

print('FOLDER NAMES:')
for f in folder_names:
    print(f)

print('-'*100)
print('JSON FOLDER NAMES')
for f in json_folder_names:
    print(f)

print("preparing dataloader...")
dataloader = DataLoader(metadata=metadata)

print("loading generator")
generator_context = Generator.from_checkpoint(
    tokenizer_file=config["generator"]["tokenizer_file"],
    checkpoint_dir=config["generator"]["checkpoint_dir"],
    pipe_parallel_size=config["generator"]["pipe_parallel_size"],
)

folder_idx = 0

print("STARTED RUN, GO TO SLEEP zzz")

with generator_context as generator:
    for conceptual_suppression_threshold in conc_sup_values:
        for suppression_factor in suppression_factor_values:

            folder_name = folder_names[folder_idx]
            result_metadata = run_eval(
                generator=generator,
                result_folder=folder_name,
                logger = logger,
                text_prompt=config["hyperparams"]["prompt_text"],
                max_batch_size=config["generator"]["max_batch_size"],
                suppression_factor=suppression_factor,
                conceptual_suppression_threshold=conceptual_suppression_threshold,
                dataloader=dataloader,
                metadata=metadata,
                use_lowercase_target=True,
                auto_decide_a_or_an=True,
                progress=True,
                raw_jsons_folder=json_folder_names[folder_idx],
                normalize=config["generator"]["normalize"],
                square_outputs=config["generator"]["square_outputs"],
		num_total_explanations=num_total_explanations
            )
            folder_idx += 1

print("RUN COMPLETE, hope you slept well :)")
