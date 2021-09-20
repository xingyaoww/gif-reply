import os


class MultilabelConfig:
    model = 'BERTweetModel'
    dataset = 'GifReplyDataset'
    multiclass = False  # do multilabel classification

    dataset_path = '/home/xingyaow/gif-reply/data/processed/dataset/bertweet-normalize/finalized-split-dataset/tweet-gif-reply.pickle'
    metadata_path = ''

    load_model_path = None  # 'checkpoints/model.pth'

    use_gpu = True  # use GPU or not
    cuda_devices = [2]  # GPU device - if list then use multiple
    num_workers = 8  # how many workers for loading data
    batch_size = 32  # batch size
    print_freq = 100  # print info every N batch

    max_epoch = 100
    lr = 1e-5  # initial learning rate
    weight_decay = 1e-3
    random_seed = 42


opt = MultilabelConfig()
if opt.use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, opt.cuda_devices))
