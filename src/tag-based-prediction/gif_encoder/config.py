import os


class MultilabelConfig:
    model = 'EfficientNetGifSeqModel'
    dataset = 'GifReplyMediaDataset'
    with_frame_seq = True

    multiclass = False
    use_seq_processor = False
    dataset_path = '/home/xingyaow/gif-reply/data/processed/dataset/twitter-gif-classification-crawled-seleted.pickle'
    metadata_path = ''
    load_model_path = None  # 'checkpoints/model.pth'

    use_gpu = True  # use GPU or not
    cuda_devices = [1]  # GPU device - if list then use multiple
    num_workers = 16  # 8  # how many workers for loading data
    prefetch_factor = 8
    batch_size = 32  # batch size
    print_freq = 100  # print info every N batch

    max_epoch = 500
    lr = 1e-5  # initial learning rate
    weight_decay = 1e-3  # 1e-1
    random_seed = 42


opt = MultilabelConfig()
if opt.use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, opt.cuda_devices))
