# Data

## GIF Reply dataset
The released GIF Reply dataset that contains 1,562,701 text-gif conversation turns can be downloaded [here](TODO).

This dataset have the following fields:
- `parent_id`: parent tweet ID
- `child_id`: child tweet ID
- `parent_text`: text of the parent tweet. NOTE: due to ? issue, we require user to get tweet text using `parent_id` through the Twitter API and then [normalize the crawled tweets](https://github.com/VinAIResearch/BERTweet#preprocess).
- `child_gif_id`: GIPHY ID of the replied GIF
- `tags`: annotated tags for the replied GIF
- `set`: whether this reply is a `train` or `test` sample

## GIF Feature dataset
The dataset is built by running [bottom-up-attention](https://github.com/airsplay/py-bottom-up-attention) and [paddleOCR](https://github.com/PaddlePaddle/PaddleOCR) to extract ROI object names, ROI features, and captions from each replied GIF in the GIF Reply dataset.

The pickled dataset can be downloaded [here](TODO).

This dataset have the following fields:
- `child_gif_id`: GIPHY ID of the replied GIF
- `gif_size`: shape of GIF
- `ocr_results`: captions extracted from four frames sampled from each quartile of the gif’s length, seperated by "[INTER_FRAME_SEP]".
- `roi_feature`: extracted ROI features from four frames sampled from each quartile of the gif’s length.
- `roi_labels`: extracted ROI labels from four frames sampled from each quartile of the gif’s length.

