# Autonomous-Driving

# Description
The Location of this competition is [autonomous-driving](https://www.kaggle.com/c/pku-autonomous-driving/overview)<br>

Who do you think hates traffic more - humans or self-driving cars? The position of nearby automobiles is a key question for autonomous vehicles ― and it's at the heart of our newest challenge.

Self-driving cars have come a long way in recent years, but they're still not flawless. Consumers and lawmakers remain wary of adoption, in part because of doubts about vehicles’ ability to accurately perceive objects in traffic.

Baidu's Robotics and Autonomous Driving Lab (RAL), along with Peking University, hopes to close the gap once and for all with this challenge. They’re providing Kagglers with more than 60,000 labeled 3D car instances from 5,277 real-world images, based on industry-grade CAD car models.

Your challenge: develop an algorithm to estimate the absolute pose of vehicles (6 degrees of freedom) from a single image in a real-world traffic environment.

Succeed and you'll help improve computer vision. That, in turn, will bring autonomous vehicles a big step closer to widespread adoption, so they can help reduce the environmental impact of our growing societies.

Please cite the following paper when using the dataset:
ApolloCar3D: A Large 3D Car Instance Understanding Benchmark for Autonomous Driving
@inproceedings{song2019apollocar3d,
title={Apollocar3d: A large 3d car instance understanding benchmark for autonomous driving},
author={Song, Xibin and Wang, Peng and Zhou, Dingfu and Zhu, Rui and Guan, Chenye and Dai, Yuchao and Su, Hao and Li, Hongdong and Yang, Ruigang},
booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
pages={5452--5462},
year={2019}
}
# Steps
1、We do some eda to analyze the data,something like distribution of cars of each model type,visualize of the labels works on the images.. <br>
2、Image preprocessing. we normalized,augment and remove some wrong to the image. <br>
3、Use Wold Coordinate and Image Coordinate to generate the centerpoint and the 4 point of the box of cars.<br>
4、we use centernet as the basebmodel for this competition, use resnet101,efficientNet-b0 and resnext101_34x4d to extract the features.<br>
5、We caculate mask_loss, regr_loss and sum all the loss with weight. The mass_loss use focal loss and regr_loss use L1_loss. <br>
6、Use AdamW as the optimizer.<br>
7、we use ensemble model as the final model whic include resnet101,efficientNet-b0 and resnext101_34x4d.<br>

# additional remarks
The content of Rotations and the Euler angles you can learn from [here](https://www.phas.ubc.ca/~berciu/TEACHING/PHYS206/LECTURES/FILES/euler.pdf)<br>

