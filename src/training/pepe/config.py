import os


class GifReplyOSCARConfig(object):
    model = 'OscarCLIPModel'
    dataset = 'GifReplyOSCARDataset'

    dataset_path = "/home/xingyaow/gif-reply/data/processed/dataset/bertweet-normalize/finalized-split-dataset/tweet-gif-reply.pickle"
    gif_feature_path = "/home/xingyaow/gif-reply/data/processed/dataset/twitter-all-gif-with-ocr-fastrcnn.pickle"
    oscar_pretrained_model_dir = "/home/xingyaow/gif-reply/src/models/MultimodalCLIP/models/base-vg-labels/ep_67_588997/"
    load_model_path = None  # 'checkpoints/model.pth'

    use_gpu = True  # use GPU or not
    cuda_devices = [2]  # GPU device - if list then use multiple
    num_workers = 8  # how many workers for loading data
    batch_size = 8 # batch size
    gif_inference_batchsize = 128

    print_freq = 100  # print info every N batch
    dcg_per_n_epoch = 3  # num of epoch trained between expensive dcg score calculation
    max_epoch = 100
    lr = 1e-6  # initial learning rate
    weight_decay = 1e-3  # 1e-1
    random_seed = 42

opt = GifReplyOSCARConfig()
if opt.use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, opt.cuda_devices))
