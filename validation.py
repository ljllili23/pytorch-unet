#evaluation the model with validation or test dataset
import os,sys
import imageio
import numpy as np
from tqdm import tqdm
import pytorch_unet
import torch
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
import torch.nn.functional as F
from torchsummary import summary

#########################
# load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "./model"
model = pytorch_unet.UNet(1)
model = model.float()
model = model.to(device)
# model.load_state_dict(torch.load(os.path.join(model_path,os.listdir(model_path)[-1])))
model.load_state_dict(torch.load("/home/leejianglee/2020_05_segmentation/pytorch-unet/model/model_1591616213.2765028.pth"))
# print("load the model {}".format(os.listdir(model_path)[-1]))
summary(model, input_size=(3, 1024, 1024))
########################


#######################
# prepare the test dataset
TEST_PATH = "../test/"
test_files = []
for dir in os.listdir(os.path.join(TEST_PATH, "images/")):
    dir = dir.split('.')
    test_files.append(dir[0])
print("the size of test_files: {}.".format(len(test_files)))

X_test = np.zeros((len(test_files), 1024, 1024, 3), dtype=np.uint8)
Y_test = np.zeros((len(test_files), 1024, 1024, 1), dtype=np.bool)

print('Getting testing data for stage 1...')
sys.stdout.flush()

sizes_test = []
for n, id_ in tqdm(enumerate(test_files), total=len(test_files)):
    img_path = TEST_PATH + '/images/' + id_ + '.png'
    img = imread(img_path)[:, :, :3]
    img = resize(img, (1024, 1024), mode='constant', preserve_range=True)
    img.shape
    X_test[n] = img

    # masks_path = TEST_PATH + '/masks/' + id_ + '.png'
    # mask = imread(masks_path)[:, :, :3]
    # mask = resize(mask, (1024, 1024, 1), mode='constant', preserve_range=True)
    # Y_test[n] = mask
##############################

model.eval()
with torch.no_grad():
    for i, input in enumerate(X_test):
        input = torch.from_numpy(input)
        input = input.unsqueeze(0)
        # print(input.shape)
        input = input.permute(0, 3, 1, 2).float().to(device)
        print(input.shape)
        pred = model(input)
        # pred =  torch.sigmoid(pred).cpu()
        pred = pred.cpu()
        pred = pred.permute(0, 2, 3, 1)
        # print(np.median(pred))
        pred = torch.squeeze(pred)
        # pred = pred.numpy()
        # pred = pred * 255.0
        # pred = (pred > 0.5)

        # pred = pred.astype('uint8')
        # print(pred.shape)
        print(test_files[i])
        imageio.imsave(os.path.join("../outputs", test_files[i]+'.png'), pred)