from dataclasses import dataclass, fields


@dataclass
class EvalParams:
    sup_fact: str = ".3"
    conc_sup: str = ".3"
    result_path: str = "/det_cos_nolog_mult"
    layers: str=None


def process_determined_hparams(hparams):
    """
    Process the 'hyperparameters:' block in the determined config. Returns a custom dataclass with the specified parameter values. Hyperparameters are optional; if provided they override the command-line args in the 'entrypoint' command; if not provided, they can also be set in 'entrypoint:' if common for all tasks/models or the default values in main.py are used.
    """

    param_mismatches = []

    fieldSet = {
        f.name if f.init else param_mismatches.append(f.name)
        for f in fields(EvalParams)
    }
    if param_mismatches != []:
        raise KeyError(
            f"The Parameters: {param_mismatches} are not included in the Dataclass."
        )

    filteredArgDict = {k: v for k, v in hparams.items() if k in fieldSet}
    print("Filtered Arg Dict", filteredArgDict)
    merged_params = EvalParams(**filteredArgDict)

    return merged_params
