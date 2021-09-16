# Quickstart for `PEPE` model
Change `dataset_path`, `gif_feature_path`, `oscar_pretrained_model_dir` in `config.py` to you local path, and then run `python3 train.py` to start training.

## Data Preperation

* Change `dataset_path ` to the path of GIF Reply Dataset. Download and prepare the dataset following instruction [here](../../../data/README.md).
* Change `gif_feature_path` to the path of GIF Feature dataset. Download and prepare the dataset following instruction [here](../../../data/README.md).

## Pretrained Oscar model
```bash
wget https://biglmdiag.blob.core.windows.net/oscar/pretrained_models/base-vg-labels.zip
unzip base-vg-labels.zip -d $MODEL_DIR
```
Change `oscar_pretrained_model_dir` to `$MODEL_DIR/ep_67_588997`.
