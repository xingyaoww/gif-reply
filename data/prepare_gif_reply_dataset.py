'''This scripts prepares the gif-reply dataset.

Required packages:
```pip install pandas twitter tqdm ujson```

Example usage:
```
    python3 prepare_gif_reply_dataset.py \
    /path/to/downloaded/gif-reply-dataset.csv \
    /path/to/processed/gif-reply-dataset.csv \
    /path/to/downloaded/gif-metadata.csv \
    /path/to/twitter_credential.json \
    /tmp/tweets.json \
```

Instructions to register Twitter Official API:

Link: https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api

After got access to the Twitter API, create `twitter_credential.json`:
```
{
	"consumer_key": "PUT YOUR KEY HERE",
	"consumer_secret": "PUT YOUR SECRET HERE",
	"access_token": "PUT YOUR KEY HERE",
	"access_secret": "PUT YOUR SECRET HERE"
}
```
'''
import argparse
import logging
import os
import random
import sys
import time
from typing import Dict
from typing import List

import pandas as pd
import twitter
import ujson as json
from tqdm import tqdm
from tweet_normalizer import normalizeTweet

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO,
)


def crawl_tweets_to_file(
    all_tweet_ids: List[str],
    tweets_output_file,
    twitter_credential_file,
):
    logging.info(f'Start crawling parent tweets to {tweets_output_file}.')
    outfile = tweets_output_file

    already_crawled = set()
    tweets_to_recrawl = set()
    if os.path.isfile(outfile) and os.path.getsize(outfile) > 0:
        with open(outfile) as f:
            for line in f:
                tweet = json.loads(line)
                already_crawled.add(tweet['id_str'])

    logging.info('Saw %d tweets to recrawl' % (len(tweets_to_recrawl)))
    tweets_to_recrawl = None

    tweet_ids = []
    for tweet_id in all_tweet_ids:
        if tweet_id not in already_crawled:
            tweet_ids.append(tweet_id)
    random.shuffle(tweet_ids)

    logging.info(
        'Saw %d new tweet IDs; already crawled %d' %
        (len(tweet_ids), len(already_crawled)),
    )

    with open(twitter_credential_file) as your_key_file:
        credentials = json.load(your_key_file)
    consumer_key = credentials['consumer_key']
    consumer_secret = credentials['consumer_secret']
    access_token = credentials['access_token']
    access_token_secret = credentials['access_secret']
    api = twitter.Api(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token_key=access_token,
        access_token_secret=access_token_secret,
        sleep_on_rate_limit=True, tweet_mode='extended',
    )

    logging.info('Start crawling tweets.')
    seen = 0
    with open(outfile, 'at') as outf:
        for i in tqdm(range(0, len(tweet_ids), 100), total=(len(tweet_ids)/100+1)):
            lookup = tweet_ids[i:i+100]
            while True:
                try:
                    tweet_jsons = api.GetStatuses(status_ids=lookup)
                    for j in tweet_jsons:
                        outf.write(j.AsJsonString() + '\n')
                        seen += 1
                    outf.flush()
                    time.sleep(1)
                    break
                except BaseException as te:
                    logging.exception(te)
                    if 'No tweet matches for specified terms' in str(te):
                        time.sleep(10)
                        break
                    time.sleep(60)
                except twitter.error.TwitterError as te2:
                    logging.exception(te2)
                    time.sleep(60)
    logging.info('Done crawling tweets.')


def get_tweet_id_to_parent_text_mapping(
    all_tweet_ids: List[str],
    tweets_output_file,
) -> Dict[str, str]:
    logging.info('Creating Tweet ID to parent text mapping.')
    all_tweet_ids = set(all_tweet_ids)
    mapping = {}
    with open(tweets_output_file) as f:
        for line in f:
            tweet = json.loads(line)
            tweet_id, parent_text = tweet['id_str'], tweet['full_text']
            if tweet_id in all_tweet_ids:
                mapping[tweet_id] = parent_text

    if len(mapping) < len(all_tweet_ids):
        logging.warning(
            f'Expecting {len(all_tweet_ids)} crawled tweets, actual {len(mapping)} crawled tweets available.'
            'This can be caused by deletion of the original tweets.',
        )
    return mapping


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_filepath', help='filepath of the downloaded gif-reply dataset.',
    )
    parser.add_argument(
        'output_filepath',
        help='filepath of the processed gif-reply dataset.',
    )
    parser.add_argument(
        'gif_metadata_filepath',
        help='filepath of the downloaded gif metadata.',
    )
    parser.add_argument(
        'twitter_credential_file',
        help='filepath for twitter credentials.',
    )
    parser.add_argument(
        'tweets_output_file',
        help='filepath for intermediate tweets json.',
    )
    args = parser.parse_args()

    # Read downloaded dataset
    dataset = pd.read_csv(args.input_filepath)
    dataset['parent_id'] = dataset['parent_id'].apply(str)
    dataset['child_id'] = dataset['child_id'].apply(str)

    # Crawl tweets using Twitter Offical API
    all_parent_ids = list(dataset['parent_id'].unique())
    crawl_tweets_to_file(
        all_parent_ids,
        tweets_output_file=args.tweets_output_file,
        twitter_credential_file=args.twitter_credential_file,
    )

    # Process `parent_text` as a field of dataset
    tweet_id_to_parent_text = get_tweet_id_to_parent_text_mapping(
        all_parent_ids, tweets_output_file=args.tweets_output_file,
    )
    dataset['parent_text'] = dataset['parent_id'].apply(
        lambda x: tweet_id_to_parent_text.get(x, ''),
    ).apply(normalizeTweet)
    logging.info('Parent text processed for each parent tweet.')

    # Match `tags` for each `child_gif_id`
    gif_id_to_tags = dict(
        pd.read_csv(args.gif_metadata_filepath)[
            ['child_gif_id', 'tags']
        ].to_numpy(),
    )
    dataset['tags'] = dataset['child_gif_id'].apply(
        lambda x: gif_id_to_tags.get(x),
    )
    logging.info('Tags processed for each reply GIF.')

    # Save processed dataset
    dataset.to_csv(args.output_filepath, index=False)
    logging.info(f'Processed dataset saved to {args.output_filepath}.')
