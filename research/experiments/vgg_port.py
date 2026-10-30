"""Faithful PyTorch port of alzheimer_cnn.h5 (frozen VGG16 -> Flatten -> Dense(256) -> Dense(4)).

Three details the notebook's old conversion got wrong, all handled here:
  1. Conv kernels: Keras stores (kh, kw, c_in, c_out); PyTorch wants (c_out, c_in, kh, kw).
  2. Flatten order: Keras flattens (N, 7, 7, 512) in height-width-channel order,
     so PyTorch's (N, 512, 7, 7) must be permuted before flattening.
  3. The head is 25088 -> 256 -> 4, not torchvision's 25088 -> 4096 -> 4096 -> 4,
     and every bias must be copied.
Input: RGB in [0, 1] (the model was trained on x / 255, without VGG16 preprocessing).
"""
import h5py
import torch
import torch.nn as nn

VGG16_BLOCKS = [2, 2, 3, 3, 3]
VGG16_WIDTHS = [64, 128, 256, 512, 512]


class KerasVGGHead(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        layers, c_in = [], 3
        for n_convs, width in zip(VGG16_BLOCKS, VGG16_WIDTHS):
            for _ in range(n_convs):
                layers += [nn.Conv2d(c_in, width, 3, padding=1), nn.ReLU(inplace=True)]
                c_in = width
            layers.append(nn.MaxPool2d(2, 2))
        self.features = nn.Sequential(*layers)
        self.fc1 = nn.Linear(7 * 7 * 512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, return_logits=False):
        f = self.features(x).permute(0, 2, 3, 1).flatten(1)  # Keras HWC flatten order
        logits = self.fc2(torch.relu(self.fc1(f)))
        return logits if return_logits else torch.softmax(logits, dim=1)


def load_from_h5(path):
    model = KerasVGGHead()
    with h5py.File(path, "r") as f:
        w = f["model_weights"]
        convs = [m for m in model.features if isinstance(m, nn.Conv2d)]
        names = [f"block{b + 1}_conv{c + 1}" for b, n in enumerate(VGG16_BLOCKS) for c in range(n)]
        with torch.no_grad():
            for conv, name in zip(convs, names):
                g = w["vgg16"][name]
                conv.weight.copy_(torch.from_numpy(g["kernel"][()]).permute(3, 2, 0, 1))
                conv.bias.copy_(torch.from_numpy(g["bias"][()]))
            for lin, name in ((model.fc1, "dense_6"), (model.fc2, "dense_7")):
                g = w[name]["sequential_3"][name]
                lin.weight.copy_(torch.from_numpy(g["kernel"][()]).T)
                lin.bias.copy_(torch.from_numpy(g["bias"][()]))
    return model.eval()
