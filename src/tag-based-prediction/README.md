# Quickstart for `tag-based` model
Following instructions below to train the `gif_classifier` and `tweet_classifier`.

1. Change `dataset_path` in `config.py` your `gif-reply-dataset.csv` path (prepare the data following instructions [here](../../data/README.md#gif-reply-dataset)).
2. Change `metadata_path` in `config.py` to your `gif-metadata.csv` path (follow instructions [here](../../data/README.md#gif-metadata)).
3. [Download](../../data/README.md#downloadall-gifs-from-twitter) GIFs file to a local directory `$GIF_PATH` if you have not done so.
4. Run `GIF_PATH=$GIF_PATH python3 train.py` to start training.

# Prerequisite
```
pip install efficientnet_pytorch transformers imageio tqdm torch pandas torchvision pandarallel pytorch_ignite numpy scikit_multilearn Pillow scikit_learn
```
