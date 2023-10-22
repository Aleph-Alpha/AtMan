# atman-open-images-eval

evaluating ATMAN's performance on the open-images-v6 segmentation dataset


# installation

1. Make sure torch, torchvision and cudatoolkit are installed and working

2. Install the Aleph-Alpha Transformer codebase [as shown here](https://gitlab.aleph-alpha.de/research/transformer#install-as-package)

2. install dependencides

```
pip install pip-tools ## optional
pip install -r requirements.txt
```

# running with generator codebase

By default, it uses this metadata: `metadata/all_classes_max_200_per_class.json`

```
python3 example_no_api.py
```

# add new requirements

Add the new requirement to requirements.in and then run:
```
pip-compile requirements.in
```