import os
from pprint import pprint

import models
import pandas as pd
import torch
from config import opt
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import data


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


writer = SummaryWriter()
set_random_seed(opt.random_seed)

# 1. Load data
train_data = getattr(data, opt.dataset)(
    opt.dataset_path, opt.gif_feature_path, train=True,
    random_state=opt.random_seed,
    oscar_pretrained_model_dir=opt.oscar_pretrained_model_dir,
)
print('Train data loaded.')
val_data = getattr(data, opt.dataset)(
    opt.dataset_path, opt.gif_feature_path, train=False,
    random_state=opt.random_seed, reuse_data=train_data,
    oscar_pretrained_model_dir=opt.oscar_pretrained_model_dir,
)
print('Validation data loaded.')
gifs_inference_dataset = data.GIFFeatureInferenceDataset(
    pd.read_pickle(opt.gif_feature_path)['child_gif_id'].to_list(),
    train_data,
)


gifs_inference_dataloader = DataLoader(
    gifs_inference_dataset,
    opt.gif_inference_batchsize,
    shuffle=False,
    num_workers=opt.num_workers,
)
collate_fn = data.get_collate_fn(opt.model)
train_dataloader = DataLoader(
    train_data,
    opt.batch_size,
    shuffle=True,
    num_workers=opt.num_workers,
    collate_fn=collate_fn,
    drop_last=True,
)
val_dataloader = DataLoader(
    val_data,
    opt.batch_size,
    shuffle=False,
    num_workers=opt.num_workers,
    collate_fn=collate_fn,
    drop_last=True,
)
print('All data loaded.')

# 2. Load model
model = getattr(models, opt.model)()
print('Model loaded.')

if opt.load_model_path:
    model.load_state_dict(torch.load(opt.load_model_path))
if opt.use_gpu:
    model.cuda()
    if len(opt.cuda_devices) > 1:
        model = torch.nn.DataParallel(model)

# 3. initialize metrics
metrics_manager = models.get_metric_class(opt.model)(
    inference_dataloader=gifs_inference_dataloader,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
)
print('Metric Manager loaded.')


def forward_step(model, inputs, labels, criterion):
    # train model
    tweet_input_ids, gif_inputs, gif_ids = inputs
    if opt.use_gpu:
        tweet_input_ids = tweet_input_ids.cuda()
        gif_inputs = [i.cuda() for i in gif_inputs]
        labels = labels.cuda()
    # with torch.cuda.amp.autocast():
    score = model(tweet_input_ids, gif_inputs)
    # Symmetric loss in CLIP paper
    loss = (criterion(score, labels) + criterion(score.T, labels)) / 2
    y_true = labels
    y_pred = score
    return loss, y_pred, y_true


def val(model, dataloader, metrics_manager, calculate_dcg=False):
    print('Validating model.')
    model.eval()

    criterion = CrossEntropyLoss()

    metrics_manager.reset()
    with torch.no_grad():
        # Disable Accuracy Metric
        for _, data in tqdm(enumerate(dataloader), total=len(val_data)//opt.batch_size):
            inputs, labels = data
            loss, y_pred, y_true = forward_step(
                model, inputs, labels, criterion=criterion,
            )
            metrics_manager.update(loss, y_pred, y_true)
        results = metrics_manager.compute(
            model=model, train=False, calculate_dcg=calculate_dcg,
        )

    model.train()
    return results


def train():
    print('Training Started.')
    # 1. Criterion and Optimizer
    lr = opt.lr
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=opt.weight_decay,
    )

    criterion = CrossEntropyLoss()  # softmax + cross-entropy

    # 2. Training start
    step = 0
    for epoch in range(opt.max_epoch):
        # Calculate DCG
        calculate_dcg = epoch % opt.dcg_per_n_epoch == 0
        # Validate
        val_results = val(
            model, val_dataloader,
            metrics_manager, calculate_dcg=calculate_dcg,
        )
        print(f'epoch {epoch} validation result: ', end='')
        pprint(val_results)

        # Tensorboard logging
        metrics_manager.log_tensorboard(
            writer, step, results=val_results, train=False,
        )

        metrics_manager.reset()
        for _, (inputs, labels) in tqdm(enumerate(train_dataloader), total=len(train_data)//opt.batch_size):
            optimizer.zero_grad()
            loss, y_pred, y_true = forward_step(
                model, inputs, labels, criterion=criterion,
            )
            loss.backward()
            optimizer.step()

            # Meters update
            metrics_manager.update(loss, y_pred, y_true)

            # Print running loss/acc
            if step % opt.print_freq == 0:
                # TensorBoard Logging
                metric_result = metrics_manager.log_tensorboard(
                    writer, step, results=None, loss=loss, train=True,
                )
                print(
                    f'step: {step}, loss: {loss.item()}',
                )

                # Training specific info
                parameter_l2_norm = torch.sum(
                    torch.stack(
                    [torch.norm(p) for p in model.parameters()], dim=0,
                    ),
                )
                gradient_l2_norm = torch.sum(
                    torch.stack(
                    [torch.norm(p.grad.data) for p in model.parameters() if p.grad is not None], dim=0,
                    ),
                )
                writer.add_scalar('parameter_l2_norm', parameter_l2_norm, step)
                writer.add_scalar('gradient_l2_norm', gradient_l2_norm, step)

            step += 1

        # Save model
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        torch.save(
            model.state_dict(),
            f'checkpoints/{opt.model}-epoch-{epoch}.pth',
        )


if __name__ == '__main__':
    train()
