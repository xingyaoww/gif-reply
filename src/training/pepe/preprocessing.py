import torch
from torch import nn
from torchvision import transforms


def pad_same_length(img: torch.Tensor):
    h, w = img.size()[-2:]
    longest = max(h, w)
    h_diff, w_diff = longest - h, longest - w
    pad_h = h_diff // 2
    pad_w = w_diff // 2
    return nn.functional.pad(img, [pad_w, w_diff - pad_w, pad_h, h_diff - pad_h])


IMAGE_MEAN = torch.tensor([0.5, 0.5, 0.5])
IMAGE_STD = torch.tensor([0.5, 0.5, 0.5])
IMAGE_SIZE = 224
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(pad_same_length),
                                transforms.ToPILImage(),
                                transforms.Resize(
                                    IMAGE_SIZE), transforms.ToTensor(),
                                transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
                                ])
