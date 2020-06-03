import os,sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import helper
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.morphology import label

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from torchsummary import summary
import torch
import torch.nn as nn
import pytorch_unet

from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss
import torch.nn as nn

import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.optim import lr_scheduler
import time
import copy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from skimage.transform import resize
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow
from skimage.transform import resize
seed = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = pytorch_unet.UNet(1)
model = model.float()
model = model.to(device)
summary(model, input_size=(3, 224, 224))

# class SimDataset(Dataset):
#     def __init__(self, count, transform=None):
#         self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.input_images)
#
#     def __getitem__(self, idx):
#         image = self.input_images[idx]
#         mask = self.target_masks[idx]
#         if self.transform:
#             image = self.transform(image)
#
#         return [image, mask]

# use same transform for train/val for this example

# trans = transforms.Compose([
#     transforms.ToTensor(),
# ])

# train_set = SimDataset(2000, transform=trans)
# val_set = SimDataset(200, transform=trans)

TRAIN_PATH = "../OLDS/data/dsb2018_96/"
TEST_PATH = "../OLDS/data/tset_dataset/"
# TEST_PATH = './stage1_test/'

# train_files = next(os.walk(TRAIN_PATH))[1]
# test_files = next(os.walk(TEST_PATH))[1]

train_files = []
for dir in os.listdir(os.path.join(TRAIN_PATH,"images/")):
    dir = dir.split('.')
    train_files.append(dir[0])

print("the size of train_files: {}.",len(train_files))

test_files = []
for dir in os.listdir(os.path.join(TEST_PATH, "images/")):
    dir = dir.split('.')
    test_files.append(dir[0])
print("the size of test_files: {}.",len(test_files))

X_train = np.zeros((len(train_files), 224, 224, 3), dtype = np.uint8)
Y_train = np.zeros((len(train_files), 224, 224, 1), dtype = np.bool)
X_test = np.zeros((len(test_files), 224, 224, 3), dtype=np.uint8)
Y_test = np.zeros((len(train_files), 224, 224, 1), dtype=np.bool)
###############################################
# get the data arrays for training;

print('Getting training data...')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_files), total=len(train_files)):
    img_path = TRAIN_PATH + '/images/' + id_ + '.png'
    img = imread(img_path)[:, :, :3]
    img = resize(img, (224, 224, 3), mode='constant', preserve_range=True)
    X_train[n] = img

    masks_path = TRAIN_PATH + '/masks/' + id_ + '.png'
    mask = imread(masks_path)[:, :, :3]
    mask = resize(mask, (224, 224, 1), mode='constant', preserve_range=True)
    Y_train[n] = mask


print('Getting testing data for stage 1...')
sys.stdout.flush()

sizes_test = []
for n, id_ in tqdm(enumerate(test_files), total=len(test_files)):
    img_path = TEST_PATH + '/images/' + id_ + '.png'
    img = imread(img_path)[:, :, :3]
    img = resize(img, (224, 224), mode='constant', preserve_range=True)
    img.shape
    X_test[n] = img

    masks_path = TEST_PATH + '/masks/' + id_ + '.png'
    mask = imread(masks_path)[:, :, :3]
    mask = resize(mask, (224, 224, 1), mode='constant', preserve_range=True)
    Y_test[n] = mask

############################################


# dataset augmentation
class Nuc_Seg(Dataset):
    def __init__(self, images_np, masks_np):
        self.images_np = images_np
        self.masks_np = masks_np

    def transform(self, image_np, mask_np):
        ToPILImage = transforms.ToPILImage()
        image = ToPILImage(image_np)
        mask = ToPILImage(mask_np.astype(np.int32))

        image = TF.pad(image, padding=20, padding_mode='reflect')
        mask = TF.pad(mask, padding=20, padding_mode='reflect')

        angle = random.uniform(-10, 10)
        width, height = image.size
        max_dx = 0.1 * width
        max_dy = 0.1 * height
        translations = (np.round(random.uniform(-max_dx, max_dx)), np.round(random.uniform(-max_dy, max_dy)))
        scale = random.uniform(0.8, 1.2)
        shear = random.uniform(-0.5, 0.5)
        image = TF.affine(image, angle=angle, translate=translations, scale=scale, shear=shear)
        mask = TF.affine(mask, angle=angle, translate=translations, scale=scale, shear=shear)

        image = TF.center_crop(image, (224, 224))
        mask = TF.center_crop(mask, (224, 224))

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

    def __len__(self):
        return len(self.images_np)

    def __getitem__(self, idx):
        image_np = self.images_np[idx]
        mask_np = self.masks_np[idx]
        image, mask = self.transform(image_np, mask_np)

        return [image, mask]


# dataloader for train and validation
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = seed)
train_set = Nuc_Seg(X_train, Y_train)
# train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
val_set = Nuc_Seg(X_val, Y_val)
# valid_loader = DataLoader(valid_dataset, batch_size = 16, shuffle = True)

image_datasets = {
    'train': train_set, 'val': val_set
}
# your dataset for training and validation;

batch_size = 25

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
}

dataset_sizes = {
    x: len(image_datasets[x]) for x in image_datasets.keys()
}

print(dataset_sizes)

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float()

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

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
num_class = 1
model = pytorch_unet.UNet(num_class).to(device)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=40)


###############
# save the model
model_path = "./model/"
if not os.path.exists(model_path):
    os.mkdir(model_path)
torch.save(model.state_dict(),os.path.join(model_path,"model_{0}.pth".format(time.time())))
print("model file saved!")