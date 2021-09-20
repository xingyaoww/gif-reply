# Quickstart for `PEPE` model
1. Change `dataset_path` in `config.py` your `gif-reply-dataset.csv` path (prepare the data following instructions [here](../../data/README.md#gif-reply-dataset))
2. Change `gif_feature_path` in `config.py` to path of the updated `gif-metadata.pkl` with additional ROI metadata (follow instructions [here](../../data/README.md#gif-metadata)).
3. Download Pretrained Oscar model to `$MODEL_DIR` (a directory of your choice):
```bash
wget https://biglmdiag.blob.core.windows.net/oscar/pretrained_models/base-vg-labels.zip
unzip base-vg-labels.zip -d $MODEL_DIR
```

4. Change `oscar_pretrained_model_dir` in `config.py` to `$MODEL_DIR/ep_67_588997`.
5. Run `python3 train.py` to start training.

# Prerequisite
```
pip install transformers torch numpy pandas pytorch_ignite pytorch_transformers pandarallel tqdm scikit_learn
```
