from interface import *
import os
from torchvision import models
from d2l import torch as d2l

net = models.resnet50(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, 100)
nn.init.xavier_uniform_(net.fc.weight)

devices, num_epochs, lr, wd = d2l.try_gpu(), 10, 2e-4, 5e-4
lr_period, lr_decay = 4, 0.1
trainloader, validloader = load_data_cifar100()
loss = nn.CrossEntropyLoss(reduction="none")
trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                          weight_decay=wd)
#train(net, trainloader, validloader, num_epochs, loss, trainer, lr_period, lr_decay)
train_fine_tuning(net, lr, lr_period, lr_decay, batch_size=128, num_epochs=5)
