# Object detection and emotion detection using Deep Learning

![image](https://user-images.githubusercontent.com/56451080/214510994-39161020-3a7c-46fc-9a39-b9a10a728afa.png)

## Object detection
For object detection(blue bounding boxes and green text-confidence level) was used pretrained FastRCNN with Resnet50 backbone

![image](https://user-images.githubusercontent.com/56451080/214513719-a60cf7ea-c25b-48ea-a2a0-51ad0a53769a.png)

## Face detection
For face detection was used OpenCV cascade classifier

<img width="256px" allign="center" src="https://user-images.githubusercontent.com/56451080/214512794-5e2dd225-ede6-4ee3-aba9-1d09210cfd57.png" />

## Emotion classification
For emotion classification was used Resnet18, that were trained on affectnet dataset for ~hour. On inference this model predicts emotion on faces that were detected by face detector (OpenCV classifier)



