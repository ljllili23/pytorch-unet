import imageio
import imgaug as ia
import os
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

aug = iaa.CropToFixedSize(width=1000, height=1000)

print("Augmentation Begins!")
if not os.path.exists("../data/aug_image"):
    os.mkdir("../data/aug_image")
if not os.path.exists("../data/aug_mask/"):
    os.mkdir("../data/aug_mask/")

image_dir = "../data/images/"
mask_dir = "../data/masks/"

for fn in os.listdir(image_dir):
    image = imageio.imread(os.path.join(image_dir,fn))
    mask = imageio.imread(os.path.join(mask_dir,fn.split('.')[0]+'.png'))
    ia.seed(13)
    for idx in range(100):
        aug_image = aug(image=image)
        imageio.imsave("../data/aug_image/{}_{}.png".format(fn.split('.')[0],idx), aug_image)
        print("../data/aug_image/{}_{}.png".format(fn,idx))
    ia.seed(13)
    for idx in range(100):
        aug_mask = aug(image=mask)
        imageio.imsave("../data/aug_mask/{}_{}.png".format(fn.split('.')[0],idx), aug_mask)
        print("../data/aug_mask/{}_{}.png".format(fn,idx))

