# GIF Reply

Official Github Repo for EMNLP 2021 paper [An animated picture says at least a thousand words: Selecting Gif-based Replies in Multimodal Dialog](https://arxiv.org/abs/2109.12212)
by [Xingyao Wang](https://xingyaoww.github.io/) and [David Jurgens](https://jurgens.people.si.umich.edu/).

![Hello Darkness my old friend GIF](https://media.giphy.com/media/p1FMSGnOdgno4/giphy.gif?cid=ecf05e47opim6j50fe72zz68gsd9wu6n4hb1vkpbrg980ndn&rid=giphy.gif&ct=g)

## Dataset
We release 1,562,701 text-gif conversation turns on Twitter, as well as the metadata of GIFs used by these conversations. More details on data can be found [here](data/README.md).

## Models
We release the code for the `PEPE` model and our baseline models (CLIP variant, Tag-based Predictions).
More details can be found below:
- `PEPE`: [code](src/pepe/), [model weight](https://drive.google.com/file/d/1fOSxCwMPGVa7LooeRemteqv45Knkcxi_/view), [colab demo](https://colab.research.google.com/drive/1pCWj6y9R_cz3tI5lsxHQtdfrTfh8pE7H)
- CLIP variant: [code](src/clip-variant)
- Tag-based Predictions: [code](src/tag-based-prediction)
