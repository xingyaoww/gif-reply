# Quickstart for `clip-variant` model
Change `dataset_path` in `config.py` to you local path, prepare the original GIF file in `.mp4` format, and then run `GIF_PATH=$GIF_PATH python3 train.py` to start training.

## Dataset Preperation
Change `dataset_path ` to the path of GIF Reply Dataset. Download and prepare the dataset following instruction [here](../../../data/README.md).

## Prepare GIF file in `$GIF_PATH` of your choice
Download all replied GIF files in `.mp4` format from Twitter, rename them with the matched GIPHY ID, and then build a 3-level directory structure that holds all these GIFs. For example, `cmzpDyYUthnOQ9uNB4.mp4` will be saved to `$GIF_PATH/c/m/z/pDyYUthnOQ9uNB4.mp4`.
