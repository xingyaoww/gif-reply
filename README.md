# Selecting a reaction GIF Reply in Multimodal Dialog

Human dialog is often multimodal. In texts and social media conversations, people frequently and often humorously use reaction gifs as direct responses to dialog. Previously, most Natural Language Processing (NLP) chatbots or dialog systems were purely text based. This repo changes that situation by introducing new computational models for selecting the appropriate reaction gif reply in conversation. 

The core task is to select a reaction gif to reply with. This task essentially creates an IR-based dialog system that, given a text message, uses a ranking function to score gifs according to their relevance as a reply to that message and then pick the most-relevant gif to reply with. 

This repo contains code for the Findings of EMNLP 2021 paper [An animated picture says at least a thousand words: Selecting Gif-based Replies in Multimodal Dialog](https://arxiv.org/abs/2109.12212)
by [Xingyao Wang](https://xingyaoww.github.io/) and [David Jurgens](https://jurgens.people.si.umich.edu/). The paper and this repo describe three models for how to implement this ranking function by training models that use (1) gif tags (i.e., annotations to gifs), (2) an [OpenAI CLIP](https://openai.com/blog/clip/) model, or (3) a multimodal OSCAR encoder. The last of those systems is known as **Pepe the King Prawn** and ultimately performed best in dataset-based evaluations and randomized controlled trial of effectiveness. Please see the paper for details.

You can read all about this project, data, model, etc in the exciting PDF technical report _or_ in the general-audience [Imgur post](https://imgur.com/gallery/G0oSrLV) that goes into more detail than it probaly should. 

## Dataset

To support work on multimodal reaction-gif based dialog, we have released a new dataset of 1,562,701 text-gif conversation turns on Twitter, as well as the metadata of GIFs used by these conversations. More details on data can be found [here](data/README.md).

## Models
We also release the code for the **Pepe the King Prawn** model and our baseline models (CLIP variant, Tag-based Predictions).
More details can be found below:
- `PEPE`: [code](src/pepe/), [model weight](https://drive.google.com/file/d/1fOSxCwMPGVa7LooeRemteqv45Knkcxi_/view), [colab demo](https://colab.research.google.com/drive/1pCWj6y9R_cz3tI5lsxHQtdfrTfh8pE7H)
- CLIP variant: [code](src/clip-variant)
- Tag-based Predictions: [code](src/tag-based-prediction)

# Interact with Pepe!

If you want to see Pepe in action before trying to work on this research problem, feel free to try Pepe out in our Slack App, which has a live Pepe model reply to messages in a Slack workspace channel of your choosing. Code and installation instructions are on the [Slack App's repo](https://github.com/xingyaoww/gif-reply-slack-bot/blob/main/README.md)

# Contact and Citation Information

For technical issues, feature requests, and bug reports, please use the Issue Tracker for this repo. For very specific questions, please contact the authors using the emails on their webpages above.

This project is based on work described in 
```
@inproceedings{
  author = {Wang, Xingyao and Jurgens, David},
  year = 2021,
  title = {{An Animated Picture Says at Least a Thousand Words: Selecting Gif-based Replies in Multimodal Dialog}},
  booktitle = {Proceedings of the Findings of the 2021 Conference on Empirical Methods in Natural Language Processing (Findings of EMNLP)}
}  
```

If you are an academic who for whatever reason needs to mention this project in your paper, please cite the above so that Xingyao will get more citations and go on to be a successful highly-cited researcher.
