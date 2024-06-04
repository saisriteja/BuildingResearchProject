
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



import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time


import wandb
from logzero import logger

def train_model(config =None):



    # with wandb.init(config=config):
    if True:

        

        # config = wandb.config
        logger.info(str(config))
        # print(config)



        from model.unet1 import ResNetUNet
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        num_class = 6
        model = ResNetUNet(num_class).to(device)

        # freeze backbone layers
        for l in model.base_layers:
            for param in l.parameters():
                param.requires_grad = False

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)




        # model, optimizer, scheduler, num_epochs=25
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        num_epochs = 25

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

        batch_size = 1

        dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
        }

        checkpoint_path = "checkpoint.pth"

        best_loss = 1e10

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            since = time.time()

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                metrics = defaultdict(float)
                epoch_samples = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = calc_loss(outputs, labels, metrics)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    epoch_samples += inputs.size(0)

                print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['loss'] / epoch_samples

                if phase == 'train':
                    scheduler.step()
                    for param_group in optimizer.param_groups:
                        print("LR", param_group['lr'])

                # save the model weights
                if phase == 'val' and epoch_loss < best_loss:
                    print(f"saving best model to {checkpoint_path}")
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), checkpoint_path)

            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        print('Best val loss: {:4f}'.format(best_loss))

        # load best model weights
        model.load_state_dict(torch.load(checkpoint_path))
        return model


if __name__ == '__main__':


    sweep_config = {
        'method': 'random'
        }

    parameters_dict = {
        'optimizer': {
            'values': ['adam', 'sgd']
            },
        'epochs': { 
            'values': [1, 2, 3]
            }
        }

    sweep_config['parameters'] = parameters_dict


    import pprint
    pprint.pprint(sweep_config)


    model = train_model(sweep_config)
    # sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
    # wandb.agent(sweep_id, train_model, count=5)
