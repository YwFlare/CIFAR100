from interface import *
from torchvision import models
from d2l import torch as d2l
from models.sknet import *
from models.resnet import *
from models.mlp_mixer import *
from models.cbam import *
from models.coordatt import *
from models.mobilenext import *
from models.triplet_att import *
from models.A2 import *
from models.non_local import *
from models.gcnet import *

# net = models.resnet50(pretrained=True)
# net.fc = nn.Linear(net.fc.in_features, 100)
# nn.init.xavier_uniform_(net.fc.weight)
# net = SKNet50(100)
# net = MLPMixer(in_channels=3, image_size=224, patch_size=16, num_classes=100,
#                dim=512, depth=8, token_dim=256, channel_dim=2048)
net = resnet50_cbam()
# net = resnet_ca50()
# net = mobilenext(num_classes=100)
# net = resnet50_da()
# net = resnet50_ta()
# net = resnet50_nl()
# net = resnet50_gc()
devices, num_epochs, lr, wd = d2l.try_gpu(), 10, 2e-4, 5e-4
lr_period, lr_decay = 4, 0.1
trainloader, validloader = load_data_cifar100()
loss = nn.CrossEntropyLoss(reduction="none")
trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
# train(net, trainloader, validloader, num_epochs, loss, trainer, lr_period, lr_decay)
train_fine_tuning(net, lr, lr_period, lr_decay, trainloader, validloader, loss, num_epochs=5)
