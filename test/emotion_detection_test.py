import torch
import torchvision
from torchvision import transforms
from torchvision.models import resnet18
import cv2
from PIL import Image

train_transform = transforms.Compose(
    [transforms.Resize(size=(224, 224)),
     transforms.ToTensor()]
)

categories = ['fearful', 'disgusted', 'angry', 'neutral', 'sad', 'surprised', 'happy']

model = resnet18()  # torch.hub.load("pytorch/vision", "resnet18", pretrained=False)
model.fc = torch.nn.Linear(512, len(categories))
model.load_state_dict(torch.load('model_064.txt'))

img = cv2.imread('data/face2.jpg')
img = Image.fromarray(img)
# gray = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

img = train_transform(img)
pred = model(img.view(1, 3, 224, 224))
print(categories[torch.argmax(pred)])