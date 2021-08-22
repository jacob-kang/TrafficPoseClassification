'''

'''
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os, glob, time, copy, random, zipfile
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import models, transforms
import torch, gc
import time

#Hyper parameters
size = 150
train_val_ratio = 0.3
batch_size = 32  # 총 파일이 M개라면, batch_size만큼으로 만들어야하므로, m/batchsize 만큼 횟수가됨.
learning_rate = 0.0001
num_epoch = 60


#Data directory
train_dir = "data\\train"  # train 시킬 사진들 디렉토리. 섞어놔도 상관없음.
test_dir = "data\\test"  # test할것글 디렉토리. 이름에 라벨이 붙어있으면 안댐.

#val_dir = "data\\val"  # train 시킬 사진들 디렉토리. 섞어놔도 상관없음.

train_list = glob.glob(os.path.join(train_dir, '*.*'))  # ../data/train/dog.890.jpg 이런 형식임.
test_list = glob.glob(os.path.join(test_dir, '*.*'))  # ../data/test/10435.jpg 이런 형식임.
#val_list = glob.glob(os.path.join(val_dir, '*.*'))  # ../data/test/10435.jpg 이런 형식임.

train_list, val_list = train_test_split(train_list, test_size=train_val_ratio,shuffle=True)

label = train_list[0].split('\\')[-1].split('_')[0]  # R or L or N임.       #이부분을 잘알아야 하는게 /와 \\가 같은의미임. 시스템에따라 달라질수있음.
print("체크중 : ",train_list[0])
print("체크중 : ",label)


classese = ['N', 'R', 'L']

class ImageTransform():

    def __init__(self):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.Resize((size, size)),  # 리사이즈
                transforms.ToTensor(),
            ])
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)


class gestureDataset(data.Dataset):

    def __init__(self, file_list, transform=None, phase=''):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        img_path = self.file_list[idx]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)

        # Get Label
        label = img_path.split('\\')[-1].split('_')[0]
        if label == 'L':  # Left
            label = 2
        elif label == 'R':  # Right
            label = 1
        elif label == 'N':  # None
            label = 0

        return img_transformed, label

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("cuda is?", torch.cuda.is_available())

# Dataset
train_dataset = gestureDataset(train_list, transform=ImageTransform(), phase='train')
val_dataset = gestureDataset(val_list, transform=ImageTransform(), phase='val')

# Operation Check
print('Operation Check1')
index = 0
print(train_dataset.__getitem__(index)[0].size())
print(train_dataset.__getitem__(index)[1])

# DataLoader
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}


# Operation Check
print('Operation Check2')
batch_iterator = iter(train_dataloader)
inputs, label = next(batch_iterator)


use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)

# Change Last Layer
# Output Features 1000 → 3
'''
(classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
'''

net.classifier[6] = nn.Linear(in_features=4096, out_features=3)

# 마지막 노드 수정
params_to_update = []

update_params_name = ['classifier.6.weight', 'classifier.6.bias']

for name, param in net.named_parameters():
    if name in update_params_name:
        param.requires_grad = True
        params_to_update.append(param)
    else:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=params_to_update, lr=learning_rate)

loss_list = []
iter_list = []
accuracy_list = []

epochLoss_list = []
epochAcc_list = []

numepoch_list = [i for i in range(1, num_epoch + 1)]


def train_model(net, dataloader_dict, criterion, optimizer, num_epoch):
    since = time.time()
    best_model_wts = copy.deepcopy(net.state_dict())
    global best_acc
    best_acc = 0.0
    net = net.to(device)

    total_iter = 1

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))

        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels in dataloader_dict[phase]:  # 이부분에서 멈춤

                inputs = inputs.to(device)
                labels = labels.to(device)  # 또 여기서 멈추는데 device가 없음.
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):  # validation이어도 들어감.
                    outputs = net(inputs)

                    _, preds = torch.max(outputs, 1)  # https://wingnim.tistory.com/34        #주어진 탠서에서 최대값을 찾는것
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item()
                    epoch_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    total_iter += 1
                    loss_list.append(loss.item())

                    accuracy_list.append(float(torch.sum(preds == labels.data) / inputs.size(0)) * 100)  # 유독 여기서 문제가터짐.
                    iter_list.append(total_iter)

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            if phase == 'train':
                epochLoss_list.append(epoch_loss * 100)
                epochAcc_list.append(epoch_acc.item() * 100)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss * 100, epoch_acc * 100))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())

    # 러닝 끝
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    net.load_state_dict(best_model_wts)
    return net


net = train_model(net, dataloader_dict, criterion, optimizer, num_epoch)

tm = time.localtime(time.time())


# Train / visualization accuracy
plt.plot(numepoch_list, epochAcc_list, color="red")
plt.xlabel("Number of Epoch")
plt.ylabel("Accuracy")
plt.title(
    "Acc. {}-{}. {}:{} size {} Batch {} Epoch {} Lr {}".format(tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min,
                                                               size, batch_size, num_epoch, learning_rate))
plt.savefig('배경합성 인조 800 Acc {} {} {}-{}'.format(tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min))
plt.clf()
# visualization loss
plt.plot(numepoch_list, epochLoss_list)
plt.xlabel("Number of Epoch")
plt.ylabel("Loss")
plt.title(
    "Loss. {}-{}. {}:{} size {} Batch {} Epoch {} Lr {}".format(tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min,
                                                                size, batch_size, num_epoch, learning_rate))
plt.savefig('배경합성 인조 800 Loss {} {} {}-{}'.format(tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min))
plt.clf()

# Prediction
id_list = []
pred_list = []
result = []
chaces = []
answer = []
L_chance = []
N_chance = []
R_chance = []

corrects = 0

L_total = 0
R_total = 0
N_total = 0

L_corrects = 0
R_corrects = 0
N_corrects = 0

with torch.no_grad():

    test_loss = 0.0

    for test_path in test_list:
        img = Image.open(test_path)
        # _id = int(test_path.split('\\')[-1].split('_')[1])
        _id = test_path.split('\\')[-1]
        _answer = test_path.split('\\')[-1].split('_')[0]

        transform = ImageTransform()
        img = transform(img,'val')
        img = img.unsqueeze(0)
        img = img.to(device)

        net.eval()

        outputs = net(img)
        preds_percents = F.softmax(outputs, dim=1)


        highest_chance = torch.max(preds_percents, 1)  # 가장 확률이 높은값
        location = highest_chance.indices[0].tolist()  # 높은값의 위치.

        if _answer == classese[location]:
            corrects += 1

        # 각 포즈별 정확도
        if _answer == 'N':
            N_total += 1

            if _answer == classese[location]:
                N_corrects += 1

        elif _answer == 'R':
            R_total += 1

            if _answer == classese[location]:
                R_corrects += 1

        elif _answer == 'L':
            L_total += 1

            if _answer == classese[location]:
                L_corrects += 1

        id_list.append(_id)
        result.append(classese[location])  # 높은값의 위치에 따른 클래스.

        N_chance.append(str(round(preds_percents.tolist()[0][0] * 100, 2)))
        R_chance.append(str(round(preds_percents.tolist()[0][1] * 100, 2)))
        L_chance.append(str(round(preds_percents.tolist()[0][2] * 100, 2)))
        answer.append(_answer)

acc = round(corrects / len(test_list) * 100, 3)

res = pd.DataFrame({
    'File id': id_list,
    'Answer': answer,
    'result': result,
    'None': N_chance,
    'Right': R_chance,
    'Left': L_chance,
    '   ': None,
    'Epoch': num_epoch,
    'Batchsize': batch_size,
    'Image Size': size,
    'Learning rate': learning_rate,
    'test acc': acc
})

res.reset_index(drop=True, inplace=True)
res.to_csv('acc {} best train {:.2f} N {:.2f} R {:.2f} L {:.2f}.csv'.format(
    acc, best_acc, (N_corrects / N_total * 100),(R_corrects / R_total * 100), (L_corrects / L_total * 100)), index=False)

# Save model

torch.save(net.state_dict(),'acc {} best train {:.2f} N {:.2f} R {:.2f} L {:.2f}.pt'.format(
    acc, best_acc, (N_corrects / N_total * 100),(R_corrects / R_total * 100), (L_corrects / L_total * 100)))