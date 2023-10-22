import yaml
# from atman_magma.magma import Magma
# from atman_magma.explainer import Explainer
from atman_magma.logit_parsing import get_delta_cross_entropies, get_delta_logits
# from atman_magma.openimages_eval import run_eval


# from multimodal_explain_eval.dataloader import DataLoader
# from multimodal_explain_eval.utils import load_json_as_dict

with open("config.yml", "r") as stream:
    config = yaml.safe_load(stream)

output_folder_root = config["files"]["output_dir"]

conceptual_suppression_threshold_values = [0.7]

suppression_factor_values = [0.1]

manipulate_attn_scores_after_scaling_values = [True, False]

modify_suppression_factor_based_on_cossim_values = [True, False]

possible_logit_parsing_functions = [get_delta_cross_entropies, get_delta_logits]

# metadata = load_json_as_dict(
#     filename = config['files']['metadata_filename']
# )
# dataloader = DataLoader(
#     metadata=metadata
# )

# print('loading model...')
# model = Magma.from_checkpoint(
#     checkpoint_path = './magma_checkpoint.pt',
#     device = 'cuda:0'
# )

output_folders = []

for conceptual_suppression_threshold in conceptual_suppression_threshold_values:
    for suppression_factor in suppression_factor_values:
        for (
            manipulate_attn_scores_after_scaling
        ) in manipulate_attn_scores_after_scaling_values:
            for (
                modify_suppression_factor_based_on_cossim
            ) in modify_suppression_factor_based_on_cossim_values:
                for logit_parsing_fn in possible_logit_parsing_functions:

                    ## x if x is not None else None
                    output_folder_name = f"{output_folder_root}/conceptual_suppression_threshold_{conceptual_suppression_threshold if conceptual_suppression_threshold is not None else None}_suppression_factor_{suppression_factor}_manipulate_attn_scores_after_scaling_{manipulate_attn_scores_after_scaling}_modify_suppression_factor_based_on_cossim_{modify_suppression_factor_based_on_cossim}_logit_parsing_fn_{logit_parsing_fn.__qualname__}"

                    print(output_folder_name)

print("eval complete :)")
