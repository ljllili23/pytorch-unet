import imageio
import imgaug as ia
import os
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

aug = iaa.CropToFixedSize(width=1000, height=1000)

print("Augmentation Begins!")
if not os.path.exists("./aug_image"):
    os.mkdir("./aug_image")
if not os.path.exists("./aug_mask/"):
    os.mkdir("./aug_mask/")

image_dir = "/home/leejianglee/2020_05_segmentation/data/images/"
mask_dir = "/home/leejianglee/2020_05_segmentation/data/masks/"

for fn in os.listdir(image_dir):
    image = imageio.imread(os.path.join(image_dir,fn))
    mask = imageio.imread(os.path.join(mask_dir,fn.split('.')[0]+'.png'))
    ia.seed(13)
    for idx in range(100):
        aug_image = aug(image=image)
        imageio.imsave("./aug_image/{}_{}.png".format(fn,idx), aug_image)
        print("./aug_image/{}_{}.png".format(fn,idx))
    ia.seed(13)
    for idx in range(100):
        aug_mask = aug(image=mask)
        imageio.imsave("./aug_mask/{}_{}.png".format(fn,idx), aug_mask)
        print("./aug_mask/{}_{}.png".format(fn,idx))

