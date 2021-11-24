# TrafficPoseRecognition_GraduationProject
**This is Traffic Pose Recognition AI for Autonomous driving.**

![ezgif com-gif-maker](https://user-images.githubusercontent.com/88817336/130382697-f722de44-a4c9-42a5-895a-6e2fbc7e6284.gif)

_Made by Jacob Jisu Kang (advisor : Prof. Haneol Jang)_

contact: dev.newjacob19@gmail.com


# Introduction

Since Tesla autopilot technique went out into the world,
The popularity of Autopitot is risng day by day.

According to SAE(Society of Automotive Engineers), Current autopilot level is 2.   
Partial automation.
<p align="center"><img src="https://user-images.githubusercontent.com/88817336/129144241-1f646283-0700-47c0-ba39-1d51152c2e32.png" width="70%" height="70%"/>

Current Autopilot doesn't care about what people are doing but cares about which direction people are going.

For example, In incident case such as traffic light malfuntion, road construction or car accident, There may most likely be someone who are traffic gestureing.
But, in current level, Autopilot may understand 'A person is blocking in front of us'.

![ezgif com-gif-maker](https://user-images.githubusercontent.com/88817336/129146787-4a205829-88d2-4e68-8db2-d82e6a4f7d07.gif)

So I'm aiming to make Traffic Pose recognition AI for understanding a person deeply.

**Note. This project is made with Pytorch. I use only Pytorch so I can't provide tesorflow version.**
  
  
# Explanations
**'AI_learning.py'** is about AI learning. You can make your own model.
This is basically configured VGG-16 (Transfer learning) and I only modified level 3 of classifer (Fully connected 3).
I have my own model (.pth file) but the size is too big to upload github due to github policy.
 
** Due to personal information, I can't provide dataset. It contains me and my family. XD. **
  
 *** 
  
**'AI_Inference_Cam.py'** is about infering from camera (Real time). fps speed is 6~7 in 1280 X 960.
  
  **Red line box** is for important ones like car, motorcycle, truck etc..
  
  **Yellow line box** is for human who is too small to detect pose or is out of ROI area.
  But when it comes closer towards camera (getting bigger in camera) at the sametime it's in ROI, it turns green or blue line box.
  
  ![ezgif com-gif-maker (2)](https://user-images.githubusercontent.com/88817336/130389629-ef7308f7-8ccf-4105-8d7e-bb44562e0142.gif)
  
  **Green line box** means this person is gestureing Left or Right.
  
  **Blue line box** means this person is gestureing None. (None of left or right). When there are some samee results of gesture after recognition in ROI, Then It shows the blue-colored result arrow on middle top.
  
  ***
  
**'AI_Inference_Pictures.py'** is inference pictures in _'inference_Pictures'_ folder.
  
  I already put some picture in that. and there is a result CSV file.
  You can see 'InferenceResult_Pictures.csv'. and in that file, how much percent did model infer those pictures.
  
  ![image](https://user-images.githubusercontent.com/88817336/130392607-43a2e619-af0a-4ad7-9fc7-66f8a5050379.png)
  ![image](https://user-images.githubusercontent.com/88817336/130392484-c0eef254-9cc8-4818-8096-ab16d0c8ce13.png)

  
  
  
  

  ***
  
**'AI_Model_Loading.py'** is about loading AI_model. You can make your own model with 'AI_learning.py' and use it.
  
  ***

**'SSD300_Model_Loading.py'** is about  loading SSD300 model.
 
  To detect object, I used SSD 300. and SSD 300 is one-stage detection. (for futher upgrade to increase speed)

  ***
  
**'category_names.txt'** is about SSD 300 categories. just to use SSD 300. Not much important.
  
  
  ***
  **Due to github policy, over 25Mb files can not be uploaded in github. Therfore, AI model and SSD300 model is not included.**
  
  
# Environment

Amd ryzen 7 2700, 16gb ram, Geforce graphic card Gtx 1660 6gb   
Anaconda, Python 3.7.1, (JetBrain) Pycharm Community 2021.1.1.   
Deep learning framework is Pytorch 1.8.0+cu111 and CUDA 11. Packages is Torchvision 0.9.0+cu111.
