# TrafficPoseRecognition_GraduationProject
**This is Traffic Pose Recognition AI for Autonomous driving.**

# Introduction

Since Tesla autopilot technique went out into the world,
The popularity of Autopitot is risng day by day.

According to SAE(Society of Automotive Engineers), Current autopilot level is 2. Partial automation.
<p align="center"><img src="https://user-images.githubusercontent.com/88817336/129144241-1f646283-0700-47c0-ba39-1d51152c2e32.png" width="70%" height="70%"/>

Current Autopilot doesn't care about what person is doing but cares about which direction person is going.

For example, In incident case such as traffic light malfuntion, road construction or car accident, There may be someone who are traffic gestureing.
But, in current level, Autopilot may understand 'a person is blocking in front of us'.

![ezgif com-gif-maker](https://user-images.githubusercontent.com/88817336/129146787-4a205829-88d2-4e68-8db2-d82e6a4f7d07.gif)

So I'm aiming to make Traffic Pose recognition AI for understanding a person deeply.

**Note. This project is made with Pytorch. I use only Pytorch so I can't provide tesorflow version.**
  
  
# Explanations
'AI_learning.py' is about AI learning. You can make your own model.
This is basically configured VGG-16 (Transfer learning) and I only modified level 3 of classifer (Fully connected 3).
I have my own model (.pth file) but the size is too big to upload github due to github policy.
  
**'AI_Inference_Cam.py'** is about inferencing form camera (Real time).
  Red line box is for important ones like car, motorcycle, truck etc..
  Yellow line box is for human who is too small to detect pose or is out of ROI area.
  
  
  
  
  But when it comes closer or bigger in camera at the sametime it's in ROI, it turns green or blue line box.
  
  
  
**'AI_Inference_Pictures.py'** is about 
  
**'AI_Model_Loading.py'** is about   

**'SSD300_Model_Loading.py'** is about 

**'category_names.txt'** is about 
