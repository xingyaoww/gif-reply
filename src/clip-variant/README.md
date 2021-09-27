# Quickstart for training the `clip-variant` model

1. Change `dataset_path` in `config.py` to you `gif-reply-dataset.csv` path (prepare the data following instructions [here](../../data/README.md#gif-reply-dataset)).
2. [Download](../../data/README.md#downloadall-gifs-from-twitter) GIFs file to a local directory `$GIF_PATH` if you have not done so.
3. Run `GIF_PATH=$GIF_PATH python3 train.py` to start training.

# Prerequisite
```
pip install imageio transformers torch tqdm torchvision pandas pandarallel numpy efficientnet_pytorch pytorch_ignite scikit_learn Pillow
```
