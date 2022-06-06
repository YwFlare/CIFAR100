from torch import nn
import torch
import torch.nn.functional as F


class DoubleAttention(nn.Module):

    def __init__(self, in_channels: int, c_m: int, c_n: int, reconstruct=False):

        super(DoubleAttention, self).__init__()
        self.c_m = c_m
        self.c_n = c_n
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.convA = nn.Conv2d(in_channels, c_m, kernel_size=1)
        self.convB = nn.Conv2d(in_channels, c_n, kernel_size=1)
        self.convV = nn.Conv2d(in_channels, c_n, kernel_size=1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):

        batch_size, c, h, w = x.size()
        assert c == self.in_channels, 'input channel not equal!'
        A = self.convA(x)  # (B, c_m, h, w) because kernel size is 1
        B = self.convB(x)  # (B, c_n, h, w)
        V = self.convV(x)  # (B, c_n, h, w)
        tmpA = A.view(batch_size, self.c_m, h * w)
        attention_maps = B.view(batch_size, self.c_n, h * w)
        attention_vectors = V.view(batch_size, self.c_n, h * w)
        attention_maps = F.softmax(attention_maps, dim=-1)  # softmax on the last dimension to create attention maps
        # step 1: feature gathering
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # (B, c_m, c_n)
        # step 2: feature distribution
        attention_vectors = F.softmax(attention_vectors, dim=1)  # (B, c_n, h * w) attention on c_n dimension
        tmpZ = global_descriptors.matmul(attention_vectors)  # B, self.c_m, h * w
        tmpZ = tmpZ.view(batch_size, self.c_m, h, w)
        if self.reconstruct: tmpZ = self.conv_reconstruct(tmpZ)
        return tmpZ


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        self.da = DoubleAttention(out_channels * BasicBlock.expansion, out_channels, out_channels, True)

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.da(self.residual_function(x)) + self.shortcut(x))


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        self.da = DoubleAttention(out_channels * BottleNeck.expansion, out_channels, out_channels, True)

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.da(self.residual_function(x)) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def resnet18_A2():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34_A2():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50_A2():
    return ResNet(BottleNeck, [3, 4, 6, 3])


def resnet101_A2():
    return ResNet(BottleNeck, [3, 4, 23, 3])


def resnet152_A2():
    return ResNet(BottleNeck, [3, 8, 36, 3])


if __name__ == '__main__':
    net = resnet50_A2()
    print(net)
