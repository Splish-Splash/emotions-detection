import cv2
import numpy as np
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import time
from torchvision.models.detection.faster_rcnn import _COMMON_META
from opencv_faces import get_faces
from PIL import Image
from torchvision import transforms
import torchvision

transform = transforms.Compose(
    [transforms.Resize(size=(224, 224)),
     transforms.ToTensor(),
     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ]
)

device = torch.device('cuda')

# emotions = np.array(['fearful', 'disgusted', 'angry', 'neutral', 'sad', 'surprised', 'happy'])
emotions = np.array(['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt'])
emotion_model = torchvision.models.resnet18()  # torch.hub.load("pytorch/vision", "resnet18", pretrained=False)
emotion_model.fc = torch.nn.Linear(512, len(emotions))
path_to_emotion_model = 'emotion_model2_0.5118.txt'
emotion_model.load_state_dict(torch.load(path_to_emotion_model))
emotion_model.to(device)


def get_emotion(face, model):
    # face = cv2.cvtColor(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2RGB)
    img = transform(Image.fromarray(face)).to(device)
    preds = model(img.view(-1, 3, 224, 224))
    return emotions[torch.argmax(preds)]


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a FastRCNN
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    hidden_layer = 256
    # and replace the mask predictor with a FastRCNN
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")  # get_model_instance_segmentation(2)
# move model to the right device
model.to(device)
categories = _COMMON_META['categories']
# model.load_state_dict(torch.load('model_params.txt'))
model.eval()
mode = 'cam'

cam = cv2.VideoCapture(1)
with torch.no_grad():
    frame_rate = 5
    prev = 0

    while True:
        # Read frame from camera if mode is cam, if not read from file
        if mode == 'cam':
            _, image_np = cam.read()
        else:
            image_np = cv2.imread('data/face1.jpg')
        time_elapsed = time.time() - prev
        if time_elapsed > 1. / frame_rate:
            prev = time.time()

            prediction = model([torch.from_numpy(image_np).to(device).div(255).permute(2, 0, 1)])[0]

            # if there's no boxes founded, display just image, else display boxes and predicted category on image
            if prediction['boxes'].shape[0] == 0:
                image_np_with_detections = image_np.copy()
            else:
                box = prediction['boxes'][0].cpu().int().numpy()
                category = categories[prediction['labels'][0].cpu().int().item()]
                confidence = prediction['scores'][0].cpu().item()

                image_np_with_detections = cv2.rectangle(image_np.copy(), box[:2], box[-2:], (255, 0, 0), 2)
                image_np_with_detections = cv2.putText(image_np_with_detections,
                                                       f'{category} {confidence * 100:.2f}%',
                                                       (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                                       0.8, (0, 255, 0), 2)
            faces, coords = get_faces(image_np)
            # display box for face and predicted emotion
            for face, (x, y, w, h) in zip(faces, coords):
                cv2.rectangle(image_np_with_detections, (x, y), (x + w, y + h), (0, 0, 255), 2)
                emotion = get_emotion(face, emotion_model)
                image_np_with_detections = cv2.putText(image_np_with_detections, f'{emotion}',
                                                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Display output
            if mode == 'cam':
                cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
            else:
                cv2.imshow('object detection', image_np_with_detections)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
