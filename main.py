from resnet import *
from interface import *

net = ResNet()
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 2e-4, 5e-4
lr_period, lr_decay = 4, 0.9
trainloader, validloader = load_data_cifar100(num_epochs)
train(net, trainloader, validloader, num_epochs, lr, wd, devices, lr_period, lr_decay)
