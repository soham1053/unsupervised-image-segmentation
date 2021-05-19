import cv2
from model import UISNet
from datasets import Rescale, ToTensor
from config import *
import torch
from torchvision import transforms
import numpy as np

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UISNet(maxPixelFeatures, nMidConvs).to(device)
model.load_state_dict(torch.load(im_folder + '.pt'))
model.eval()

rescale = Rescale((height, width))
to_tensor = ToTensor()

label_colours = np.random.randint(255,size=(maxPixelFeatures,3))
def plot_segmented(output):
    output = output.permute(1, 2, 0).contiguous().view(-1, maxPixelFeatures)
    _, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[c % maxPixelFeatures] for c in im_target])
    im_target_rgb = im_target_rgb.reshape((height, width, 3)).astype(np.uint8)
    cv2.imshow('Output', im_target_rgb)

while True:
    ret, frame = cap.read()
    frame = rescale(frame[:, ::-1, :])
    cv2.imshow('Input', frame)

    frame = to_tensor(frame).unsqueeze(0).to(device)
    segmented = model(frame)[0]
    plot_segmented(segmented)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()