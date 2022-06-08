from interface import *
from models.resnet import *
from models.sknet import *
from models.mlp_mixer import *
from models.cbam import *
from models.coordatt import *
from models.mobilenext import *
from models.triplet_att import *
from models.A2 import *
from models.non_local import *
from models.gcnet import *
from models.mixcs import *
from models.mixsc import *

model = [
    'A2',
    'cbam',
    'coordatt',
    'gcnet',
    'mlp_mixer',
    'mobilenext',
    'non_local',
    'resnet',
    'sknet',
    'triplet_att',
    'mixsc'
]

A2 = resnet50_A2()
cbam = resnet50_cbam()
coordatt = resnet50_coordatt()
gcnet = resnet50_gc()
mlp_mixer = MLPMixer(in_channels=3, dim=256, token_mix=128, channel_mix=1024, img_size=32, patch_size=4, depth=8,
                     num_classes=100)
mob = mobilenext(num_classes=100)
non_local = resnet50_nl()
resnet = resnet50()
sknet = SKNet50()
triplet_att = resnet50_ta()
mixsc = resnet50_sc()

nets = [A2, cbam, coordatt, gcnet, mlp_mixer, mob, non_local, resnet, sknet, triplet_att, mixsc]
device = d2l.try_gpu()
trainloader, validloader = load_data_cifar100()
for i, net in enumerate(nets):
    model_path = 'pth/' + model[i] + '.pth'
    net.load_state_dict(torch.load(model_path))
    net.to(device)
    acc = evaluate_accuracy(net, validloader, device)
    print(model[i] + ' acc: ' + str(acc))
