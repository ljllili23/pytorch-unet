#evaluation the model with validation or test dataset
import os,sys
import numpy as np
from tqdm import tqdm
import pytorch_unet
import torch
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
from torchsummary import summary

#########################
# load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "./model"
model = pytorch_unet.UNet(1)
model = model.float()
model = model.to(device)
model.load_state_dict(torch.load(os.path.join(model_path,os.listdir(model_path)[0])))
print("load the model {}".format(os.listdir(model_path)[0]))
summary(model, input_size=(3, 224, 224))
########################


#######################
# prepare the test dataset
TEST_PATH = "../OLDS/data/tset_dataset/"
test_files = []
for dir in os.listdir(os.path.join(TEST_PATH, "images/")):
    dir = dir.split('.')
    test_files.append(dir[0])
print("the size of test_files: {}.".format(len(test_files)))

X_test = np.zeros((len(test_files), 224, 224, 3), dtype=np.uint8)
Y_test = np.zeros((len(test_files), 224, 224, 1), dtype=np.bool)

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
##############################

model.eval()
with torch.no_grad():
    for i, input in enumerate(X_test):
        input = torch.from_numpy(input)
        input = input.unsqueeze(0)
        # print(input.shape)
        input = input.permute(0, 3, 1, 2).float().to(device)
        # print(input.shape)
        pred = model(input)
        pred = pred.permute(0, 2, 3, 1)
        pred = pred.cpu()
        # pred = (pred > 0.5)
        pred = torch.squeeze(pred)
        # print(pred.shape)
        print(test_files[i])
        plt.imsave(os.path.join(os.getcwd(), "outputs", test_files[i]), pred)