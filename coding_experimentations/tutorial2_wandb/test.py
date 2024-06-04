import math


from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader
from data.data_utils import SimDataset

from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss
from loss_utils.loss import calc_loss

import time
import sys
from utils.metrics_printer import print_metrics
import torch

from utils.dataset_check import reverse_transform


import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time

from model.unet1 import ResNetUNet

import sys
sys.path.append('/home/saiteja/Desktop/Paper_Works/BuildingResearchProject/coding_experimentations/tutorial1_setup/pytorch-unet')
import helper


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_class = 6
model = ResNetUNet(num_class).to(device)

# use the same transformations for train/val in this example
trans = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
])

model.eval()   # Set model to the evaluation mode

# Create a new simulation dataset for testing
test_dataset = SimDataset(3, transform = trans)
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)

# Get the first batch
inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.to(device)
print('inputs.shape', inputs.shape)
print('labels.shape', labels.shape)

# Predict
pred = model(inputs)
# The loss functions include the sigmoid function.
pred = torch.sigmoid(pred)
pred = pred.data.cpu().numpy()
print('pred.shape', pred.shape)

# Change channel-order and make 3 channels for matplot
input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

# Map each channel (i.e. class) to each color
target_masks_rgb = [helper.masks_to_colorimg(x) for x in labels.cpu().numpy()]
pred_rgb = [helper.masks_to_colorimg(x) for x in pred]
helper.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])