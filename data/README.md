# Data

## GIF Reply dataset
### Download
The released GIF Reply dataset `gif-reply-dataset.csv` ([download here](TODO)) that contains 1,562,701 text-gif conversation turns.

This dataset has the following fields:
- `parent_id`: parent tweet ID
- `child_id`: child tweet ID
- `child_gif_id`: hash ID of the replied GIF (hash for a GIF can be calculated using `python3 data/hash_gif.py`)
- `set`: whether this reply is a `train` or `test` sample

This downloaded dataset lacks serveral data fields for model training, following the instructions below to prepare the complete version of GIF reply dataset.
### Preparation
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
    /tmp/tweets.json \
```

## GIF metadata
### Download
This dataset ([download here](TODO)) contains metadata for GIFs.

It has the following fields:
- `child_gif_id`: hash ID of the replied GIF
- `ocr_text`: captions extracted using [paddleOCR](https://github.com/PaddlePaddle/PaddleOCR) from four frames sampled from each quartile of the gif’s length, seperated by "[INTER_FRAME_SEP]". 
- `tags`: annotated tags for GIF

### Additional ROI metadata
ROI metadata is only required to train the `PEPE` model. 
Additional ROI metadata can be extracted using [bottom-up-attention](https://github.com/airsplay/py-bottom-up-attention).

Preparation script to extract ROIs can be found [here (forthcomming)](TODO), and it will add the following two fields of data.
- `roi_feature`: extracted ROI features on four frames sampled from each quartile of the gif’s length.
- `roi_labels`: extracted ROI labels on four frames sampled from each quartile of the gif’s length.


## GIF url mapping
### Download
This data ([download here](TODO)) contains a mapping from the GIF ID (hash ID) to a url of that GIF hosted on GIPHY.

It has the following fields:
- `gif_id`: hash ID of a GIF
- `giphy_url`: url of that GIF hosted on [GIPHY](https://giphy.com/)
