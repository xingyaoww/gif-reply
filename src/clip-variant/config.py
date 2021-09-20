import os


class GifReplyConfig:
    model = 'CLIPModel'
    dataset = 'GifReplyDataset'

    dataset_path = 'FILL YOUR PATH TO gif-reply-dataset.csv HERE'
    load_model_path = None  # 'checkpoints/model.pth'

    use_gpu = True  # use GPU or not
    cuda_devices = [0]  # GPU device - if list then use multiple
    num_workers = 8  # how many workers for loading data
    batch_size = 16  # batch size
    gif_inference_batchsize = 64

    print_freq = 100  # print info every N batch
    dcg_per_n_epoch = 3  # num of epoch trained between expensive dcg score calculation
    max_epoch = 100
    lr = 1e-5  # initial learning rate
    weight_decay = 1e-3
    random_seed = 42


opt = GifReplyConfig()
if opt.use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, opt.cuda_devices))
