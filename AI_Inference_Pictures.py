'''
이건 model이 test셋에서 얼마나 성능을 내느냐를 체크하기위함..
'''

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, glob, time, copy, random, zipfile
from PIL import Image
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import models, transforms
import torch, gc
import time
from AI_Model_Loading import net

size = 150


test_dir = "inference_Pictures"
test_list = glob.glob(os.path.join(test_dir, '*.*'))          #../data/test/10435.jpg 이런 형식임.

classese = ['N','R','L']

class ImageTransform():

    def __init__(self):
        self.data_transform = {
            'test': transforms.Compose([
                transforms.Resize((size,size)),
                transforms.ToTensor(),
            ])
        }
    def __call__(self, img, phase):
        return self.data_transform[phase](img)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("cuda is?",torch.cuda.is_available())


# Prediction
id_list = []
pred_list = []
result = []
chaces = []
answer = []
L_chance = []
N_chance = []
R_chance = []

corrects=0

tm=time.localtime(time.time())

with torch.no_grad():
    for test_path in test_list:
        img = Image.open(test_path)
        _id = test_path

        transform = ImageTransform()
        img = transform(img, phase='test')
        img = img.unsqueeze(0)
        img = img.to(device)

        net.eval()

        outputs = net(img)
        preds_percents = F.softmax(outputs, dim=1)
        highest_chance = torch.max(preds_percents, 1)       #가장 확률이 높은값
        location =highest_chance.indices[0].tolist()        #높은값의 위치.

        id_list.append(_id)
        result.append(classese[location])                   #높은값의 위치에 따른 클래스.

        N_chance.append(str(round(preds_percents.tolist()[0][0]*100,2)))
        R_chance.append(str(round(preds_percents.tolist()[0][1]*100,2)))
        L_chance.append(str(round(preds_percents.tolist()[0][2]*100,2)))

acc=round(corrects/len(test_list) *100,3)

res = pd.DataFrame({
    'File id': id_list,
    'result':result,
    'None':N_chance,
    'Right':R_chance,
    'Left':L_chance,
    '   ':None,
})

res.reset_index(drop=True, inplace=True)

res.to_csv('InferenceResult_Pictures.csv', index=False)