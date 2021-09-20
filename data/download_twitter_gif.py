
'''This scripts downloads

Required packages:
```pip install pandas tqdm requests```

Example usage:
```
    python3 download_twitter_gif.py \
    /path/to/downloaded/gif-twitter-url-mapping.csv \
    /path/to/store/downloaded/gifs/
```
'''

import argparse
import pandas as pd
import sys
import os
import time
import logging
import requests
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)


parser = argparse.ArgumentParser()
parser.add_argument("gif_url_mapping_filepath", help="filepath of the downloaded gif-twitter url mapping.")
parser.add_argument("output_gif_dir", help="filepath to store the downloaded GIFs.")
args = parser.parse_args()

gif_url_mapping = pd.read_csv(args.gif_url_mapping_filepath)

def gif_id_to_filepath(gif_id: str, basepath=args.output_gif_dir, ext='.mp4') -> str:
    def _gif_id_to_structured_path(gif_id):
        return os.path.join(gif_id[0], gif_id[1], gif_id[2], gif_id[3:])
    return os.path.join(basepath, _gif_id_to_structured_path(gif_id)+ext)


def download_gif_file(gif_id: str, download_url: str):
    # Try to download gif, If success, add filepath to database
    gif_filepath = gif_id_to_filepath(gif_id)
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
            logging.error(f"Failed to download GIF file: {download_url}")
    except Exception as e:
        logging.exception(e)
        gif_filepath = None
        try:
            os.remove(gif_filepath)
        except:
            pass

    return gif_filepath

if __name__ == "__main__":
    for i, row in tqdm(gif_url_mapping.iterrows(), total=len(gif_url_mapping)):
        gif_id = row["gif_id"] # hash id
        twitter_url = row["twitter_url"]
        download_gif_file(gif_id, twitter_url)
