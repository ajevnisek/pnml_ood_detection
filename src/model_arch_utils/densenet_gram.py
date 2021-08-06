import collections
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_arch_utils.gram_model_utils import gram_record


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, inter_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(
            inter_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate

        self.collecting = False
        self.gram_feats = []

    def forward(self, x):

        out = self.conv1(self.relu(self.bn1(x)))
        self.record(out)

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)

        out = self.conv2(self.relu(self.bn2(out)))
        self.record(out)

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

    def record(self, t):
        feature = gram_record(t, self.collecting)
        self.gram_feats.append(feature)

    def reset(self):
        self.gram_feats = []


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.droprate = dropRate

        self.collecting = False
        self.gram_feats = []

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        self.record(out)

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

    def record(self, t):
        feature = gram_record(t, self.collecting)
        self.gram_feats.append(feature)

    def reset(self):
        self.gram_feats = []


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, growth_rate, nb_layers, dropRate
        )

        self.collecting = False
        self.gram_feats = []

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(in_planes + i * growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        t = self.layer(x)
        self.record(t)
        return t

    def record(self, t):
        feature = gram_record(t, self.collecting)
        self.gram_feats.append(feature)

    def reset(self):
        self.gram_feats = []


class DenseNet3Gram(nn.Module):
    def __init__(
        self,
        depth,
        num_classes,
        growth_rate=12,
        reduction=0.5,
        bottleneck=True,
        dropRate=0.0,
    ):
        super(DenseNet3Gram, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n / 2
            block = BottleneckBlock
        else:
            block = BasicBlock
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(
            3, in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans1 = TransitionBlock(
            in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate
        )
        in_planes = int(math.floor(in_planes * reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans2 = TransitionBlock(
            in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate
        )
        in_planes = int(math.floor(in_planes * reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.output = self.fc
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)

        # Added this method for pNML ood detection
        self.features_out = out.clone()
        return self.fc(out)

    def get_features(self):
        return self.features_out

    def load(self, path="../models/densenet_cifar10.pth"):
        tm = torch.load(path, map_location="cpu")
        if isinstance(tm, collections.OrderedDict):
            self.load_state_dict(tm, strict=False)
        else:
            self.load_state_dict(tm.state_dict(), strict=False)

    def set_collecting(self, is_collecting: bool):
        for layer in [self.block1, self.trans1, self.block2, self.trans2, self.block3]:
            if isinstance(layer, DenseBlock):
                for layer_i in layer.modules():
                    if isinstance(layer_i, BottleneckBlock):
                        layer_i.collecting = is_collecting
            layer.collecting = is_collecting

    def reset(self):
        for layer in [self.block1, self.trans1, self.block2, self.trans2, self.block3]:
            if isinstance(layer, DenseBlock):
                for layer_i in layer.modules():
                    if isinstance(layer_i, BottleneckBlock):
                        layer_i.reset()
            layer.reset()

    def collect_gram_features(self):
        gram_feats_all = []
        for layer in [self.block1, self.trans1, self.block2, self.trans2, self.block3]:
            if isinstance(layer, DenseBlock):
                for layer_i in layer.modules():
                    if isinstance(layer_i, BottleneckBlock):
                        gram_feats = layer_i.gram_feats
                        gram_feats_all += gram_feats
            gram_feats = layer.gram_feats
            gram_feats_all += gram_feats
        self.reset()
        return gram_feats_all


if __name__ == "__main__":
    torch_model = DenseNet3Gram(100, num_classes=10)
    torch_model.set_collecting(True)

    # Test
    batch = torch.empty((1, 3, 32, 32))
    _ = torch_model(batch)
    gram_features = torch_model.collect_gram_features()
    print(len(gram_features))
