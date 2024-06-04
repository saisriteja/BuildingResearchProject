import torchvision.utils
import numpy as np
from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader
from data.data_utils import SimDataset
from matplotlib import pyplot as plt



def reverse_transform(inp):
  inp = inp.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1)
  inp = (inp * 255).astype(np.uint8)

  return inp





if __name__ == '__main__':

    # use the same transformations for train/val in this example
    trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])

    train_set = SimDataset(2000, transform = trans)
    val_set = SimDataset(200, transform = trans)

    image_datasets = {
    'train': train_set, 'val': val_set
    }

    batch_size = 25

    dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    # Get a batch of training data
    inputs, masks = next(iter(dataloaders['train']))

    print(inputs.shape, masks.shape)

    plt.imshow(reverse_transform(inputs[3]))