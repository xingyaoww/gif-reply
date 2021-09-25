'''This scripts downloads GIFs from Twitter.

Required packages:
```pip install pandas tqdm requests twitter```

Example usage:
```
python3 download_twitter_gif.py \
    /path/to/downloaded/gif-reply-dataset.csv \
    /path/to/twitter_credential.json \
    /path/to/cache/child-tweets.json \
    /path/to/store/downloaded/gifs/ # this is the $GIF_PATH of your choice
```
'''
import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from prepare_gif_reply_dataset import crawl_tweets_to_file
from tqdm import tqdm


logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO,
)


def gif_id_to_filepath(gif_id: str, gif_dir, ext='.mp4') -> str:
    def _gif_id_to_structured_path(gif_id):
        return os.path.join(gif_id[0], gif_id[1], gif_id[2], gif_id[3:])
    return os.path.join(gif_dir, _gif_id_to_structured_path(gif_id)+ext)


def download_gif_file(gif_id: str, download_url: str, gif_dir: str):
    # Try to download gif, If success, add filepath to database
    gif_filepath = gif_id_to_filepath(gif_id, gif_dir=gif_dir)
    # Make dir if not exists
    Path(os.path.dirname(gif_filepath)).mkdir(parents=True, exist_ok=True)

    if os.path.exists(gif_filepath):
        # skipping gif if already downloaded
        return gif_filepath

    try:
        img_file = requests.get(download_url)
        # If download success
        if img_file.status_code == 200:
            # Store the gif file
            with open(gif_filepath, 'wb') as f:
                f.write(img_file.content)
        else:
            gif_filepath = None
    except Exception as e:
        logging.error(f'Exception occurred for {twitter_url}')
        logging.exception(e)
        gif_filepath = None
        try:
            os.remove(gif_filepath)
        except:
            pass

    return gif_filepath


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'gif_reply_dataset_filepath',
        help='filepath of the downloaded gif-reply dataset.',
    )
    parser.add_argument(
        'twitter_credential_file',
        help='filepath for twitter credentials.',
    )
    parser.add_argument(
        'tweets_output_file',
        help='filepath for intermediate tweets json.',
    )
    parser.add_argument(
        'output_gif_dir', help='filepath to store the downloaded GIFs.',
    )
    args = parser.parse_args()

    # Read reply dataset and build child_gif_id -> [child_id1, child_id2, ...] mapping
    dataset = pd.read_csv(args.gif_reply_dataset_filepath)
    dataset['parent_id'] = dataset['parent_id'].apply(str)
    dataset['child_id'] = dataset['child_id'].apply(str)
    gif_id_with_child_id = dataset[['child_gif_id', 'child_id']].groupby(
        'child_gif_id',
    )['child_id'].apply(list).reset_index()

    # Crawl child tweets using Twitter Offical API
    all_child_ids = list(dataset['child_id'].unique())
    crawl_tweets_to_file(
        all_child_ids, args.tweets_output_file, args.twitter_credential_file,
    )

    # Read crawled child tweets and build tweet_id -> gif_url mapping
    child_tweets = pd.read_json(
        args.tweets_output_file, orient='records', lines=True,
    )
    child_tweets['id'] = child_tweets['id'].apply(str)

    def _get_url(media):
        try:
            return media[0]['video_info']['variants'][0]['url']
        except:
            return None

    child_tweets['gif_url'] = child_tweets['media'].apply(_get_url)
    tweet_id_to_animated_gif_url = dict(
        child_tweets[['id', 'gif_url']].to_numpy(),
    )

    # Download every GIF
    for i, row in tqdm(gif_id_with_child_id.iterrows(), total=len(gif_id_with_child_id)):
        gif_id = row['child_gif_id']  # hash id
        child_ids = row['child_id']

        res = ''
        # attempt for all available child gif id
        for child_id in child_ids:
            twitter_url = tweet_id_to_animated_gif_url.get(child_id)
            if twitter_url is None:
                continue
            res = download_gif_file(gif_id, twitter_url, args.output_gif_dir)
            if res is not None:
                break
        if res is None:
            # Note that some of the GIFs hosted on Twitter might be removed by the original poster.
            logging.error(f'Failed to download GIF file for ID {gif_id}')
