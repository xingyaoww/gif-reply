import os
from pprint import pprint

import models
import torch
from config import opt
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import data


def _get_attr_from_opt(opt, attribute_name, default_value):
    return getattr(opt, attribute_name) if hasattr(
        opt, attribute_name,
    ) else default_value


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


writer = SummaryWriter()
set_random_seed(opt.random_seed)

# 1. Load data
train_data = getattr(data, opt.dataset)(
    opt.dataset_path, opt.metadata_path, train=True, multiclass=opt.multiclass,
    random_state=opt.random_seed,
    with_frame_seq=_get_attr_from_opt(opt, 'with_frame_seq', False),
    image_size=_get_attr_from_opt(opt, 'image_size', 228),
)
val_data = getattr(data, opt.dataset)(
    opt.dataset_path, opt.metadata_path, train=False, multiclass=opt.multiclass,
    random_state=opt.random_seed, reuse_data=train_data,
    with_frame_seq=_get_attr_from_opt(opt, 'with_frame_seq', False),
    image_size=_get_attr_from_opt(opt, 'image_size', 228),
)

collate_fn = getattr(data, opt.collate_fn) if hasattr(
    opt, 'collate_fn',
) else None
train_dataloader = DataLoader(
    train_data,
    opt.batch_size,
    shuffle=True,
    num_workers=opt.num_workers,
    collate_fn=collate_fn,
)
val_dataloader = DataLoader(
    val_data,
    opt.batch_size,
    shuffle=False,
    num_workers=opt.num_workers,
    collate_fn=collate_fn,
)

# 2. Load model
model = getattr(models, opt.model)(
    num_classes=train_data.num_classes,
    use_seq_processor=_get_attr_from_opt(
        opt, 'use_seq_processor', True,
    ),
)
scaler = torch.cuda.amp.GradScaler()

if opt.load_model_path:
    def remove_module_prefix(state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('module'):
                new_key = '.'.join(key.split('.')[1:])
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict
    print(f'Loading checkpoint {opt.load_model_path}')
    _state_dict = remove_module_prefix(torch.load(opt.load_model_path))
    model.load_state_dict(_state_dict)
    print(f'Loaded weights from {opt.load_model_path}')
if opt.use_gpu:
    model.cuda()
    if len(opt.cuda_devices) > 1:
        model = torch.nn.DataParallel(model)

# 3. initialize metrics
metrics_manager = models.get_metric_class(opt.model)(multiclass=opt.multiclass)


def val(model, dataloader, metrics_manager):
    model.eval()

    if opt.multiclass:
        criterion = CrossEntropyLoss()
    else:
        criterion = BCEWithLogitsLoss()

    metrics_manager.reset()
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            inputs, labels = data
            with torch.cuda.amp.autocast():
                if opt.use_gpu:
                    inputs = inputs.cuda()
                    labels = [i.cuda() for i in labels] if type(
                        labels,
                    ) == list else labels.cuda()
                    score = model(inputs)
                    loss = [criterion(score, labels)]
        metrics_manager.update(scaler.scale(loss), score, labels)

    results = metrics_manager.compute()

    model.train()
    return results


def train():
    # 1. Criterion and Optimizer
    lr = opt.lr
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=opt.weight_decay,
    )

    if opt.multiclass:
        criterion = CrossEntropyLoss()  # softmax + cross-entropy
    else:
        criterion = BCEWithLogitsLoss()  # sigmoid + cross-entropy

    # 2. Training start
    step = 0
    for epoch in range(opt.max_epoch):
        metrics_manager.reset()
        for _, (inputs, labels) in tqdm(enumerate(train_dataloader), total=len(train_data)//opt.batch_size):
            # train model
            if opt.use_gpu:
                inputs = inputs.cuda()
                labels = [i.cuda() for i in labels] if type(
                    labels,
                ) == list else labels.cuda()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                score = model(inputs)
                loss = [criterion(score, labels)]
                sum(scaler.scale(loss)).backward()
                scaler.step(optimizer)
                scaler.update()
            # Meters update
            metrics_manager.update(loss, score, labels)

            # Print running loss/acc
            if step != 0 and step % opt.print_freq == 0:
                # TensorBoard Logging
                metric_result = metrics_manager.log_tensorboard(
                    writer, step, results=None, loss=loss, train=True,
                )
                print(
                    f'step: {step}, loss: {loss[0].item()}',
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

        # Validate
        val_results = val(model, val_dataloader, metrics_manager)
        print(f'epoch {epoch} validation result: ', end='')
        pprint(val_results)

        # Tensorboard logging
        metrics_manager.log_tensorboard(
            writer, step, results=val_results, train=False,
        )
        metrics_manager.log_tensorboard(
            writer, step, results=None, loss=loss, train=True,
        )

        # Save model
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        torch.save(
            model.state_dict(),
            os.path.join('checkpoints', f'{opt.model}-epoch-{epoch}.pth'),
        )


if __name__ == '__main__':
    train()
