import torch
import torch.nn
import torch.nn.functional as F
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock


class MultiHeadMLP(torch.nn.Module):
    def __init__(self, x_dim, h_dim, y_dim, n_heads=1, shared_h_dims=(100, 100)):
        super().__init__()
        # check for bad parameters
        if n_heads < 1:
            raise ValueError('Network requires at least one head.')

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.n_heads = n_heads

        shared_dims = [x_dim] + list(shared_h_dims) + [h_dim]

        # list of layers in shared network
        self.shared_layers = torch.nn.ModuleList([
            torch.nn.Linear(shared_dims[i], shared_dims[i + 1]) for i in
            range(len(shared_dims) - 1)
        ])

        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(self.h_dim, self.y_dim) for _ in range(n_heads)
        ])

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, head_idx):
        for layer in self.shared_layers:
            x = F.relu(layer(x))

        x = self.heads[head_idx](x)

        x = self.softmax(x)
        return x

    def prediction(self, x, head_idx):
        return torch.argmax(self(x, head_idx), dim=1)

    def point_estimate_loss(self, x, y, head_idx):
        predictions = self(x, head_idx)
        loss = torch.nn.NLLLoss()(predictions, y)
        return loss

    def vcl_loss(self, x, y, task_length, head_idx):
        self.point_estimate_loss(x, y, head_idx)

    def reset_for_new_task(self):
        pass


class Encoder(torch.nn.Module):
    def __init__(self, x_dim, z_dim, h_dims=(500,)):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.out_dim = 2  # output parameters for a normal

        layer_dims = [x_dim] + list(h_dims) + [z_dim]

        # list of layers in shared network
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(layer_dims[i], layer_dims[i + 1]) for i in range(len(layer_dims) - 1)
        ])

    def forward(self, x, head_idx):
        z = x
        for layer in self.layers[:-1]:
            z = F.relu(layer(z))
        z = self.layers[-1](z)

        return z

    def encode(self, x, head_idx):
        pass

    def reset_for_new_task(self):
        for layer in self.layers:
            layer.reset_parameters()


class MLPClassifier(torch.nn.Module):
    """ A simple MLP neural network for image classification """
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.linear_1 = torch.nn.Linear(x_dim, 512)
        self.linear_2 = torch.nn.Linear(512, 256)
        self.linear_3 = torch.nn.Linear(256, 128)
        self.linear_4 = torch.nn.Linear(128, 64)
        self.linear_5 = torch.nn.Linear(64, y_dim)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        h = F.relu(self.linear_1(x))
        h = F.relu(self.linear_2(h))
        h = F.relu(self.linear_3(h))
        h = F.relu(self.linear_4(h))
        h = self.linear_5(h)
        return h

    def predict(self, x):
        preds = self.softmax(self(x))
        return torch.argmax(preds, dim=1)


class Conv2DClassifier(torch.nn.Module):
    """ A simple convolutional neural network for image classification """
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels, out_channels=16, kernel_size=3)
        self.conv_2 = torch.nn.Conv2d(16, out_channels=32, kernel_size=3)
        self.max_pool = torch.nn.MaxPool2d((2, 2))
        self.linear_1 = torch.nn.Linear(32 * 12 * 12, n_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        h = F.relu(self.conv_1(x))
        h = F.relu(self.conv_2(h))
        h = F.relu(self.max_pool(h))
        h = h.view(-1, 32 * 12 * 12)
        h = self.softmax(self.linear_1(h))
        return h

    def predict(self, x):
        h = self(x)
        h = torch.argmax(h, dim=1)
        return h


# All credits for the MNIST-adapted ResNet model go to:
# https://zablo.net/blog/post/using-resnet-for-mnist-in-pytorch-tutorial/index.html
class MnistResNet(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64,
                                     kernel_size=(7, 7),
                                     stride=(2, 2),
                                     padding=(3, 3), bias=False)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        up_sampled = F.interpolate(x, size=(224, 224))
        return self.softmax(super(MnistResNet, self).forward(up_sampled))
