from PIL import Image
import os
import cv2
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import numpy as np
from AI_Model_Loading import net, device, GestureClass,model_name
import torch.nn.functional as F
import time

from SSD300_Model_Loading import nvidia_ssd_processing_utils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# SSD 모델
class ResNet(nn.Module):
    def __init__(self, backbone='resnet50', backbone_path=None):
        super().__init__()
        if backbone == 'resnet18':
            backbone = resnet18(pretrained=not backbone_path)
            self.out_channels = [256, 512, 512, 256, 256, 128]
        elif backbone == 'resnet34':
            backbone = resnet34(pretrained=not backbone_path)
            self.out_channels = [256, 512, 512, 256, 256, 256]
        elif backbone == 'resnet50':
            backbone = resnet50(pretrained=not backbone_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == 'resnet101':
            backbone = resnet101(pretrained=not backbone_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        else:  # backbone == 'resnet152':
            backbone = resnet152(pretrained=not backbone_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x
class SSD300(nn.Module):
    def __init__(self, backbone=ResNet('resnet50')):
        super().__init__()

        self.feature_extractor = backbone

        self.label_num = 81  # number of COCO classes
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.loc = []
        self.conf = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(oc, nd * self.label_num, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self._init_weights()

    def _build_additional_features(self, input_size):
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(
                zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.label_num, -1)))

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, x):
        x = self.feature_extractor(x)

        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        return locs, confs
class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """

    def __init__(self, dboxes):
        super(Loss, self).__init__()
        self.scale_xy = 1.0 / dboxes.scale_xy
        self.scale_wh = 1.0 / dboxes.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduce=False)
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0),
                                   requires_grad=False)
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.con_loss = nn.CrossEntropyLoss(reduce=False)

    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, ]
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels
            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """
        mask = glabel > 0
        pos_num = mask.sum(dim=1)

        vec_gd = self._loc_vec(gloc)

        # sum on four coordinates, and mask
        sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)
        sl1 = (mask.float() * sl1).sum(dim=1)

        # hard negative mining
        con = self.con_loss(plabel, glabel)

        # postive mask will never selected
        con_neg = con.clone()
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # number of negative three times positive
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        # print(con.shape, mask.shape, neg_mask.shape)
        closs = (con * (mask.float() + neg_mask.float())).sum(dim=1)

        # avoid no object detected
        total_loss = sl1 + closs
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss * num_mask / pos_num).mean(dim=0)
        return ret

# 텐서화
class ImageTransform():
    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.Resize((size, size)),  # 리사이즈
            transforms.ToTensor(),
        ])

    def __call__(self, img):
        return self.data_transform(img)

# SSD모델 파라미터 불러오기.
ssd_model = SSD300()
ssd_model.load_state_dict(torch.load('ssd300.pth'))
ssd_model.to('cuda')
ssd_model.eval()

if torch.cuda.is_available():
    print("Cuda mode Online")
else:
    print("****** Warning!!  Cuda mode Offline")

utils = nvidia_ssd_processing_utils()

class_names = open("category_names.txt").readlines()
class_names = [c.strip() for c in class_names]
classes_to_labels = class_names

GestureTypeClass = ['None', 'Right', 'Left']


#Video input
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # width=1920
cap.set(4, 1080)  # height=1080
#1920 1080이 최대인데, 1080 1080으로 하면 다운사이징이 아니라 centerCrop임.

print(cap.get(cv2.CAP_PROP_FPS))
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


#Video output
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#out = cv2.VideoWriter('{} {} AI demo {}frame.avi'.format(Video_name.split('.')[0],model_name,int(cap.get(cv2.CAP_PROP_FPS))), fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(3)), int(cap.get(4))))
#PoseFrameCount = int(cap.get(cv2.CAP_PROP_FPS))


#Parameters
size = 150

SSD_confidence = 40         #SSD가 몇퍼센트를 기준으로 그 이상의 결과가 나오면 물체로 볼것인지. ex)어떤 물체를 봣는데 Confidence의 %기준으로 인식할지 말지 결정.
No_drawing = False #결과값 그리기.

PoseFrameCount = 8
Frame_list = [None for i in range(PoseFrameCount)]      #8개의 프레임을 기준으로 특정포즈 결과가 일부% 이상을 차지하면 결과 출력.


Image_w ,Image_h = int(cap.get(3)),int(cap.get(4))
frameCount = 0
timer = 1


while cap.isOpened():  # 영상 시작
    success, frame = cap.read()

    if success:
        start = time.time()
        Input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Input이미지를 1:1비율로 만들기
        longer = int(max(Input_image.size))
        filledImage = Image.new("RGB", (longer, longer))
        filledImage.paste(Input_image)  # 0,0 기준으로 붙임.


        # Todo 디버깅용
        #filledImage.save("Sqaured Image.jpg")

        image = filledImage
        OriginalImage = Image.fromarray(frame)

        # Image -> ndarray
        ndarrayImage = np.array(image)

        print(type(ndarrayImage))
        print(ndarrayImage.shape)

        inputs = [utils.prepare_input(ndarrayImage)]  # 여기서 300 300 3으로 바뀜. 또한 List로 들어가야함. #파일의 이름이 들어감.
        tensor = utils.prepare_tensor(inputs)

        with torch.no_grad():
            detections_batch = ssd_model(tensor)

        try:
            results_per_input = utils.decode_results(detections_batch)
        except:
            continue

        best_results_per_input = [utils.pick_best(results, SSD_confidence / 100) for results in
                                  results_per_input]  # 이부분에서 Confidence 조정.

        for image_idx in range(len(best_results_per_input)):
            filledImage = image

            image_min = min(image.size)

            image = inputs[image_idx] / 2 + 0.5

            img = np.array(OriginalImage)

            bboxes, classes, confidences = best_results_per_input[image_idx]  # 0번배열에 물체의 박스위치, 1번에 물체의 카테고리, 2번에 컨피던스.

            for idx in range(len(bboxes)):
                left, bot, right, top = bboxes[idx]
                local_positions = [left, bot, right, top]

                x, y, w, h = [int(val * max(OriginalImage.size)) for val in [left, bot, right - left, top - bot]]
                # x는 왼쪽위 x점. y는 왼쪽위 y점. h는 높이. w는 넓이.

                for i, v in enumerate(local_positions):
                    if v > 1:
                        local_positions[i] = 1

                local_positions = [v * image_min for v in local_positions]
                # 왼쪽위 구석을 0,0으로 기준으로 두고, 거기서 왼쪽부분은 Left고 오른쪽부분은 right고 위는 bot이고 아래는 top임. 그래서 bot보다 top이 큼.

                if classes[idx] == 1:
                    local_positions = [local_positions[0] - 10, local_positions[1] - 10, local_positions[2] + 10,
                                   local_positions[3] + 10]
                else:
                    local_positions = [local_positions[0], local_positions[1], local_positions[2],
                                       local_positions[3]]


                # 왼쪽, 위 ,오른쪽 아래
                # print(local_positions)

                for i, v in enumerate(local_positions):
                    if v < 0:
                        local_positions[i] = 0

                if (classes[idx] >=2 and classes[idx] <=8) and classes[idx] != 5:
                    if not No_drawing :
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)

                        cv2.putText(img, "{}".format(classes_to_labels[classes[idx] - 1]), (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
                        cv2.putText(img, "{:.2f}%".format(confidences[idx] * 100),
                                    (x, y + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)

                        cv2.putText(img, "{}".format(classes_to_labels[classes[idx] - 1]), (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 2)
                        cv2.putText(img, "{:.2f}%".format(confidences[idx] * 100),
                                    (x, y + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 2)



                elif classes[idx] == 1:     #사람이라면??
                    if local_positions[3] - local_positions[1] > 180 and confidences[idx] > 0.6:           #영상 크기에 맞게 반응형으로 바꿔야함

                        timer = 0

                        #여기서부터는 인식도 잘되고 어느정도 수신호라고 볼수있음. 수신호 인식 시작.
                        PersonRect = filledImage.crop(
                            (filledImage.size[0] / 2 - image_min / 2, 0, filledImage.size[0] / 2 + image_min / 2,
                             image_min))
                        PersonRectImage = PersonRect.crop((local_positions))

                        # Todo 디버깅용      이미지 저장
                        #PersonRectImage.save('debug/human picture {}.jpg'.format(frameCount))
                        frameCount += 1

                        transform = ImageTransform()

                        GestureInputImage = transform(PersonRectImage)
                        GestureInputImage = GestureInputImage.unsqueeze(0)
                        GestureInputImage = GestureInputImage.to(device)

                        outputs = net(GestureInputImage)

                        preds_percents = F.softmax(outputs, dim=1)

                        highest_chance = torch.max(preds_percents, 1)  # 가장 확률이 높은값
                        location = highest_chance.indices[0].tolist()  # 높은값의 위치.


                        if highest_chance.values[0] >= 0.38:        #38%이상의 정확도가 나와야 유의미한 결과로 결정.

                            # Frame_List에 값 넣어서 평균 확인하기.
                            if None in Frame_list:  # 한자리라도 비어있으면,
                                for i in range(len(Frame_list)):
                                    if Frame_list[i] == None:
                                        Frame_list[i] = GestureClass[location]
                                        break
                            else:  # 꽉 차있으면
                                del Frame_list[0]
                                Frame_list.append(GestureClass[location])

                        if not No_drawing:

                            BGR_white = (255, 255, 255)
                            BGR_Green = (0, 255, 0)

                            BGR_Array = [BGR_white, BGR_white, BGR_white]

                            BGR_Array[location] = BGR_Green

                            if GestureClass[location] == 'N':       #N이면 파란색, L과 R은 초록색
                                cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0),
                                              2)  # (top,left),(bot,right),

                            elif GestureClass[location] == 'L':
                                cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0),
                                              2)  # (top,left),(bot,right),
                            else:
                                cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0),
                                              2)  # (top,left),(bot,right),

                            cv2.putText(img, 'Person {:.2f}%'.format(confidences[idx] * 100), (x, y-15),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0,0,0), 3)
                            cv2.putText(img, 'Person {:.2f}%'.format(confidences[idx] * 100), (x, y-15),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, BGR_white, 2)


                            cv2.putText(img, 'Left {:.2f}%'.format(preds_percents.tolist()[0][2] * 100), (x, y + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                            cv2.putText(img, 'Right {:.2f}%'.format(preds_percents.tolist()[0][1] * 100), (x, y + 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                            cv2.putText(img, 'None {:.2f}%'.format(preds_percents.tolist()[0][0] * 100), (x, y + 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)

                            cv2.putText(img, 'Left {:.2f}%'.format(preds_percents.tolist()[0][2] * 100), (x, y + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BGR_Array[2], 2)
                            cv2.putText(img, 'Right {:.2f}%'.format(preds_percents.tolist()[0][1] * 100), (x, y + 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BGR_Array[1], 2)
                            cv2.putText(img, 'None {:.2f}%'.format(preds_percents.tolist()[0][0] * 100), (x, y + 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BGR_Array[0], 2)

                        PersonRectImage = np.array(PersonRectImage)
                        PersonRectImage = cv2.cvtColor(PersonRectImage, cv2.COLOR_BGR2RGB)


                        if local_positions[3] - local_positions[1] > 200:           #영상 크기에 맞게 반응형으로 바꿔야함
                            #여기는 100퍼센트 수신호임. 그래서 결과를 내줘야함..

                            #만약 이 크기가 갑자기 어디서 나타났다고 가정하면, 화살표를 내주지못한다..
                            if Frame_list.count(None) < len(Frame_list)*0.4 and Frame_list.count('N') <len(Frame_list)*0.4:

                                if max(Frame_list, key=Frame_list.count) == 'R':
                                    cv2.arrowedLine(img, (int(Image_w / 2) - 75, 50), (int(Image_w / 2) + 75, 50),
                                                    (255, 255, 255), 9)
                                    cv2.arrowedLine(img, (int(Image_w/2) - 75, 50), (int(Image_w/2)+75, 50), (255, 0, 0), 8)

                                elif max(Frame_list, key=Frame_list.count) == 'L':
                                    cv2.arrowedLine(img, (int(Image_w / 2) + 75, 50), (int(Image_w / 2) - 75, 50),
                                                    (255, 255, 255), 9)
                                    cv2.arrowedLine(img, (int(Image_w/2)+75, 50), (int(Image_w/2) - 75, 50), (255, 0, 0), 8)

                                if not No_drawing:
                                    rows, cols, channels = PersonRectImage.shape
                                    PersonRectImage = cv2.addWeighted(img[0:0 + rows, 0:0 + cols], 0, PersonRectImage, 1, 0)

                                    img[0:0 + rows, 0:0 + cols] = PersonRectImage
                            else:
                                print("Frame_list에 None이 너무 많음. 의사결정 불가")


                    else:       #사람의 크기가 일정크기보다 작다면 그냥 행인.
                        if not No_drawing:
                            cv2.rectangle(img, (x, y), (x + w+10, y + h+10), (0, 255, 255), 2)

                            cv2.putText(img, "{}".format(classes_to_labels[classes[idx] - 1]), (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
                            cv2.putText(img, "{:.2f}%".format(confidences[idx] * 100),
                                        (x, y + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)

                            cv2.putText(img, "{}".format(classes_to_labels[classes[idx] - 1]), (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 2)
                            cv2.putText(img, "{:.2f}%".format(confidences[idx] * 100),
                                        (x, y + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 2)

                else:       #사람과 차가 아니라면
                    if not No_drawing:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

                        cv2.putText(img, "{}".format(classes_to_labels[classes[idx] - 1]), (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
                        cv2.putText(img, "{:.2f}%".format(confidences[idx] * 100),
                                    (x, y + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)

                        cv2.putText(img, "{}".format(classes_to_labels[classes[idx]-1]), (x, y -10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 2)
                        cv2.putText(img, "{:.2f}%".format(confidences[idx] * 100),
                                    (x, y + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 2)
        end = time.time()
        cv2.putText(img, "fps : {}".format(int(1/(end-start))),(int(cap.get(3)-100), 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        #out.write(img)
        cv2.imshow("test",img)
        cv2.waitKey(1)

    else:
        cap.release()
        #out.release()
        cv2.destroyAllWindows()
        print("끝")
        break