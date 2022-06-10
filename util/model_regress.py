import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Transition(nn.Sequential):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.add_module("nrom", nn.BatchNorm1d(num_features=input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv1d(in_channels=input_features,
                        out_channels=output_features, kernel_size=1, stride=1, padding=0, bias=False))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))


class DenseLayer(nn.Sequential):
    def __init__(self, input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super().__init__()
        self.add_module("norm1", nn.BatchNorm1d(num_features=input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv1d(in_channels=input_features,
                        out_channels=growth_rate * bn_size, kernel_size=1, stride=1, padding=1, bias=False))

        self.add_module("norm2", nn.BatchNorm1d(
            num_features=growth_rate * bn_size))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv1d(in_channels=growth_rate * bn_size,
                        out_channels=growth_rate, kernel_size=3, stride=1, padding=0, bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # print([m.shape for m in inputs])
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(
            self.relu1(self.norm1(concated_features)))

        return bottleneck_output

    def forward(self, input):
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(inputs=prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)

        return new_features


class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate,
                               growth_rate, bn_size, drop_rate, memory_efficient)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, features):
        features = [features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

#default growth_rate = 32
class DenseNet(nn.Module):
    def __init__(self, growth_rate=16, block_config=(6, 12, 24, 16), num_input_features=1, num_init_features=64, bn_size=4, drop_rate=0, num_classes=10000, memory_efficient=False):
        super().__init__()

        # 1. First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(num_input_features, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm1d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # 2 - 6. DenseBlock
            block = DenseBlock(num_layers, num_features,
                               bn_size, growth_rate, drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                # 3 - 6. Transition
                trans = Transition(num_features, num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        # 7
        self.features.add_module('norm5', nn.BatchNorm1d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool1d(out, (1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class Net_encoder(nn.Module):
    def __init__(self, input_size):
        super(Net_encoder, self).__init__()
        self.input_size = input_size
        self.k = 224
        self.f = 224

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 224)
        )

    def forward(self, data):
        data = data.float().view(-1, self.input_size)
        embedding = self.encoder(data)

        return embedding


class Net_cell(nn.Module):
    def __init__(self, num_of_class):
        super(Net_cell, self).__init__()
        self.cell = DenseNet(num_classes=num_of_class)

    def forward(self, embedding):
        cell_prediction = self.cell(embedding.view(embedding.shape[0], 1, -1))

        return cell_prediction