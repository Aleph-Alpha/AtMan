import json
import requests
import numpy as np
import matplotlib.pyplot as plt

def complete(
    api_token,
    prompt="Markus likes steak",
    maximum_tokens=20,
    model="luminous-extended",
    url="https://test.api.aleph-alpha.com/complete",
):

    payload = json.dumps(
        {"model": model, "prompt": prompt, "maximum_tokens": maximum_tokens}
    )

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, data=payload)
    return response.json()


def explain(
    api_token,
    prompt="Markus likes steak",
    target=" and fries",
    directional=False,
    suppression_factor=0.1,
    conceptual_suppression_threshold=0.8,
    model="luminous-extended",
    url="https://api.aleph-alpha.com/explain",
):

    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "target": target,
            "directional": directional,
            "suppression_factor": suppression_factor,
            "conceptual_suppression_threshold": conceptual_suppression_threshold,
        }
    )

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, data=payload)
    return response.json()


def visualize_result_text(
    result,
    target_token_index=0,
    prompt_start_index=None,
    prompt_end_index=None,
    width=20,
    height=10,
    topk=None,
    fontsize=22,
    save_as="output.jpg",
    xticks_rotation=0,
):

    selected_result = result["result"][target_token_index]
    explanations = selected_result["explanations"][prompt_start_index:prompt_end_index]
    target_token = selected_result["target_token_str"]

    assert (
        len(explanations) != 0
    ), "length of explanations is 0, you probably cropped it too much"

    prompt_tokens = [
        f'{i}:{explanations[i]["token_str"]}' for i in range(len(explanations))
    ]

    values = [explanations[i]["value"] for i in range(len(explanations))]

    if topk is not None:
        indices = np.argsort(np.array(values)).astype(int)[::-1]
        values = np.take(values, indices)[:topk]
        prompt_tokens = np.take(prompt_tokens, indices)[:topk]

    fig = plt.figure(figsize=(width, height))
    plt.title(f"target: {target_token}", fontsize=fontsize)
    plt.bar(
        prompt_tokens,
        values,
    )
    plt.tick_params(axis="x", rotation=xticks_rotation)
    plt.xticks(fontsize=fontsize)
    plt.grid()

    fig.savefig(save_as)
