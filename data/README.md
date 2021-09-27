# Data

## GIF Reply dataset
### Download
The released GIF Reply dataset `gif-reply-dataset.csv` ([download here](https://drive.google.com/file/d/1GClR5KLOsYAgYSS3iKP1k6-qcynR7d7g/view?usp=sharing)) that contains 1,562,701 text-gif conversation turns.

This dataset has the following fields:
- `parent_id`: parent tweet ID
- `child_id`: child tweet ID
- `child_gif_id`: hash ID of the replied GIF (hash for a GIF can be calculated using `python3 data/hash_gif.py`)
- `set`: whether this reply is a `train` or `test` sample

NOTE: This downloaded dataset lacks serveral data fields for model training, following the instructions below to prepare the complete version of GIF reply dataset.

### Prepare the `gif-reply-dataset.csv`
First, [Get access to the Twitter Official API](https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api).

After obtained access to Twitter API, create `twitter_credential.json` and filling your credentials in that file:
```
{
	"consumer_key": "PUT YOUR KEY HERE",
	"consumer_secret": "PUT YOUR SECRET HERE",
	"access_token": "PUT YOUR KEY HERE",
	"access_secret": "PUT YOUR SECRET HERE"
}
```

And then install required packages:
```
pip install pandas twitter tqdm ujson
```

Finally, run the [preparation script](prepare_gif_reply_dataset.py) to prepare the dataset:
```
python3 prepare_gif_reply_dataset.py \
    /path/to/downloaded/gif-reply-dataset.csv \
    /path/to/processed/gif-reply-dataset.csv \
    /path/to/downloaded/gif-metadata.csv \
    /path/to/twitter_credential.json \
    /tmp/tweets.json
```

### Download all GIFs from Twitter
All GIFs need to be downloaded for the purpose of model training. Note that some of the GIFs hosted on Twitter might not be available at download time, as they can be removed by the original poster.

GIFs can be downloaded by using [this](download_twitter_gif.py) script:
```
python3 download_twitter_gif.py \
    /path/to/downloaded/gif-reply-dataset.csv \
    /path/to/twitter_credential.json \
    /path/to/cache/child-tweets.json \
    /path/to/store/downloaded/gifs/ # this is the $GIF_PATH of your choice
```

This script will download all replied GIF files in `.mp4` format from Twitter, and then store them in a 3-level directory structure under `$GIF_PATH`.
For example, the GIF with hash ID `68e460404503373feee6f1c686007078dec7c0c602026667` will be saved to `$GIF_PATH/6/8/e/460404503373feee6f1c686007078dec7c0c602026667.mp4`.

## GIF metadata
### Download
This dataset `gif-metadata.csv` ([download here](https://drive.google.com/file/d/1GClR5KLOsYAgYSS3iKP1k6-qcynR7d7g/view?usp=sharing)) contains metadata for GIFs.

It has the following fields:
- `gif_id`: hash ID of the replied GIF
- `ocr_text`: captions extracted using [paddleOCR](https://github.com/PaddlePaddle/PaddleOCR) on four frames sampled from each quartile of the gif’s length, seperated by `"[INTER_FRAME_SEP]"`.
- `tags`: annotated tags for GIF. This is NOT needed to reproduce the `PEPE` model, and is only provided for replicability of the tag-based model.

### Additional ROI metadata
ROI metadata is only required to train the `PEPE` model.
Additional ROI metadata can be extracted using [bottom-up-attention](https://github.com/airsplay/py-bottom-up-attention).

Preparation script to extract ROIs can be found [here (forthcomming)](TODO), and it will add the following two fields of data into a updated pickle file `gif-metadata-with-roi.pkl`.
- `roi_feature`: extracted ROI features on four frames sampled from each quartile of the gif’s length.
- `roi_labels`: extracted ROI labels on four frames sampled from each quartile of the gif’s length.


## GIF GIPHY mapping
### Download

The file `gif-id-to-giphy-id-mapping.csv` ([download here](https://drive.google.com/file/d/1wadTg8qJGZWD6YR37xzuyXTJVSEtHx5X/view?usp=sharing)) contains a mapping from the GIF ID (hash ID) to GIPHY ID.

`giphy-id-to-gif-id-mapping.csv` ([download here](https://drive.google.com/file/d/1wadTg8qJGZWD6YR37xzuyXTJVSEtHx5X/view?usp=sharing)) contains a mapping fron the GIPHY ID to the GIF ID (hash ID).


Both files have the following fields:
- `gif_id`: hash ID of a GIF
- `giphy_id`: ID of the matched GIF on [GIPHY](https://giphy.com/). The GIF hosted on GIPHY can be found using link `https://giphy.com/gifs/[GIPHY_ID_HERE]`.


## Inferred GIF Feature for `PEPE`

The file `gif-pepe-inferred-features.csv` ([download here](https://drive.google.com/file/d/1GClR5KLOsYAgYSS3iKP1k6-qcynR7d7g/view?usp=sharing)) contains inferred GIF feature from a trained `PEPE` model for 115,404 GIFs.

It has the following fields:
- `gif_id`: hash ID of a GIF
- `gif_feature`: a feature embedding for the corresponding gif.

`gif-pepe-inferred-features-top1k.csv` ([download here](https://drive.google.com/file/d/1GClR5KLOsYAgYSS3iKP1k6-qcynR7d7g/view?usp=sharing)) is a subset of features for 1000 most used GIFs, and is provided for demonstration purpose.

## `PEPE` model checkpoint

The model checkpoint `PEPE-model-checkpoint.pth` can be [download here](https://drive.google.com/file/d/1fOSxCwMPGVa7LooeRemteqv45Knkcxi_/view), further on how to load the checkpoint can be found in the demo [here](TODO).
