import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
import torch.nn.functional as F

use_pretrained = False
net = models.vgg16(pretrained=use_pretrained)

#device 선택
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#마지막 노드 수정
net.classifier[6] = nn.Linear(in_features=4096, out_features=3)
params_to_update = []

update_params_name = ['classifier.6.weight', 'classifier.6.bias']
for name, param in net.named_parameters():
    if name in update_params_name:
        param.requires_grad = True
        params_to_update.append(param)
    else:
        param.requires_grad = False

GestureClass = ['N','R','L']

model_name = '교차검증 800인조 60에폭 배치 32 Lr 0.0001 8 14 6-17 fold5.pt'

net.load_state_dict(torch.load(model_name))
net = net.to(device)
net.eval()