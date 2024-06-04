
import sys
sys.path.append('/home/saiteja/Desktop/Paper_Works/BuildingResearchProject/coding_experimentations/tutorial1_setup/pytorch-unet')

import matplotlib.pyplot as plt
import numpy as np
import helper
import simulation


# Generate some random images
def get_sample_datset():
    input_images, target_masks = simulation.generate_random_data(192, 192, count=3)

    print("input_images shape and range", input_images.shape, input_images.min(), input_images.max())
    print("target_masks shape and range", target_masks.shape, target_masks.min(), target_masks.max())

# Change channel-order and make 3 channels for matplot
    input_images_rgb = [x.astype(np.uint8) for x in input_images]

# Map each channel (i.e. class) to each color
    target_masks_rgb = [helper.masks_to_colorimg(x) for x in target_masks]


    return input_images, target_masks, input_images_rgb, target_masks_rgb






from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

class SimDataset(Dataset):
  def __init__(self, count, transform=None):
    self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)
    self.transform = transform

  def __len__(self):
    return len(self.input_images)

  def __getitem__(self, idx):
    image = self.input_images[idx]
    mask = self.target_masks[idx]
    if self.transform:
      image = self.transform(image)

    return [image, mask]




if __name__  == '__main__':
    input_images, target_masks, input_images_rgb, target_masks_rgb = get_sample_datset()
    helper.plot_side_by_side([input_images_rgb, target_masks_rgb])