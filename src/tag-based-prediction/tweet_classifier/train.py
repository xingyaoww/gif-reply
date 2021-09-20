from config import opt
from tqdm import tqdm
import models
import data
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pprint import pprint
import os
import torch


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
)
print("Train data loaded.")
val_data = getattr(data, opt.dataset)(
    opt.dataset_path, opt.metadata_path, train=False, multiclass=opt.multiclass,
    random_state=opt.random_seed, reuse_data=train_data,
)
print("Validation data loaded.")

collate_fn = data.get_collate_fn(opt.model)
train_dataloader = DataLoader(train_data,
                              opt.batch_size,
                              shuffle=True,
                              num_workers=opt.num_workers,
                              collate_fn=collate_fn,
                              )
val_dataloader = DataLoader(val_data,
                            opt.batch_size,
                            shuffle=False,
                            num_workers=opt.num_workers,
                            collate_fn=collate_fn,
                            )
print("All data loaded.")

# 2. Load model
model = getattr(models, opt.model)(num_classes=train_data.num_classes)
print("Model loaded.")

if opt.load_model_path:
    model.load_state_dict(torch.load(opt.load_model_path))
if opt.use_gpu:
    model.cuda()
    if len(opt.cuda_devices) > 1:
        model = torch.nn.DataParallel(model)

# 3. initialize metrics
metrics_manager = models.get_metric_class(opt.model)(multiclass=opt.multiclass)
print("Metric Manager loaded.")


def forward_step(model, inputs, labels, criterion):
    # train model
    if opt.use_gpu:
        inputs = inputs.cuda()
        if isinstance(labels, list) and isinstance(labels[0], tuple):
            # expected: [(regression_gif_feature : Tensor, gif_ids : list), classification labels]
            labels[0] = (labels[0][0].cuda(), labels[0][1])
            labels[1] = labels[1].cuda()
        else:
            labels = [i.cuda() for i in labels] if type(
                labels) == list else labels.cuda()
        score = model(inputs)
        loss = [criterion(score, labels)]
        y_true = labels
        y_pred = score
    return loss, y_pred, y_true


def val(model, dataloader, metrics_manager):
    model.eval()

    if opt.multiclass:
        criterion = CrossEntropyLoss()
    else:
        criterion = BCEWithLogitsLoss()

    metrics_manager.reset()
    for _, data in enumerate(dataloader):
        inputs, labels = data
        loss, y_pred, y_true = forward_step(
            model, inputs, labels, criterion=criterion)
        metrics_manager.update(loss, y_pred, y_true)

    results = metrics_manager.compute(train=False)

    model.train()
    return results


def train():
    print("Training Started.")
    # 1. Criterion and Optimizer
    lr = opt.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=opt.weight_decay)

    if opt.multiclass:
        criterion = CrossEntropyLoss()  # softmax + cross-entropy
    else:
        criterion = BCEWithLogitsLoss()  # sigmoid + cross-entropy

    # 2. Training start
    step = 0
    for epoch in range(opt.max_epoch):
        metrics_manager.reset()
        for _, (inputs, labels) in tqdm(enumerate(train_dataloader), total=len(train_data)//opt.batch_size):
            optimizer.zero_grad()
            loss, y_pred, y_true = forward_step(
                model, inputs, labels, criterion=criterion)
            sum(loss).backward()
            optimizer.step()

            # Meters update
            metrics_manager.update(loss, y_pred, y_true)

            # Print running loss/acc
            if step % opt.print_freq == 0:
                # TensorBoard Logging
                metric_result = metrics_manager.log_tensorboard(
                    writer, step, results=None, loss=loss, train=True)
                print(
                    f"step: {step}, loss: {loss[0].item()}")

                # Training specific info
                parameter_l2_norm = torch.sum(torch.stack(
                    [torch.norm(p) for p in model.parameters()], dim=0))
                gradient_l2_norm = torch.sum(torch.stack(
                    [torch.norm(p.grad.data) for p in model.parameters() if p.grad is not None], dim=0))
                writer.add_scalar("parameter_l2_norm", parameter_l2_norm, step)
                writer.add_scalar("gradient_l2_norm", gradient_l2_norm, step)

            step += 1

        # Validate
        val_results = val(model, val_dataloader, metrics_manager)
        print(f"epoch {epoch} validation result: ", end='')
        pprint(val_results)

        # Tensorboard logging
        metrics_manager.log_tensorboard(
            writer, step, results=val_results, train=False)

        # Save model
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        torch.save(model.state_dict(),
                   f"checkpoints/{opt.model}-epoch-{epoch}.pth")


if __name__ == '__main__':
    train()
