import imageio
import imgaug as ia
import os
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

aug = iaa.CropToFixedSize(width=512, height=512)

print("Augmentation Begins!")
if not os.path.exists("../data/aug_image"):
    os.mkdir("../data/aug_image")
if not os.path.exists("../data/aug_mask/"):
    os.mkdir("../data/aug_mask/")
if not os.path.exists("../data/validation/aug_image"):
    os.mkdir("../data/validation/aug_image")
if not os.path.exists("../data/validation/aug_mask/"):
    os.mkdir("../data/validation/aug_mask/")


image_dir = "../data/images/"
mask_dir = "../data/masks/"
val_img_dir = "../data/validation/images/"
val_mask_dir = "../data/validation/masks/"

# for fn in os.listdir(image_dir):
#     image = imageio.imread(os.path.join(image_dir,fn))
#     mask = imageio.imread(os.path.join(mask_dir,fn.split('.')[0]+'.png'))
#     ia.seed(13)
#     for idx in range(100):
#         aug_image = aug(image=image)
#         imageio.imsave("../data/aug_image/{}_{}.png".format(fn.split('.')[0],idx), aug_image)
#         print("../data/aug_image/{}_{}.png".format(fn,idx))
#     ia.seed(13)
#     for idx in range(100):
#         aug_mask = aug(image=mask)
#         imageio.imsave("../data/aug_mask/{}_{}.png".format(fn.split('.')[0],idx), aug_mask)
#         print("../data/aug_mask/{}_{}.png".format(fn,idx))


for fn in os.listdir(val_img_dir):
    image = imageio.imread(os.path.join(val_img_dir,fn))
    mask = imageio.imread(os.path.join(val_mask_dir,fn.split('.')[0]+'.png'))
    ia.seed(13)
    for idx in range(100):
        aug_image = aug(image=image)
        imageio.imsave("../data/validation/aug_image/{}_{}.png".format(fn.split('.')[0],idx), aug_image)
        print("../data/validation/aug_image/{}_{}.png".format(fn.split('.')[0],idx))
    ia.seed(13)
    for idx in range(100):
        aug_mask = aug(image=mask)
        imageio.imsave("../data/validation/aug_mask/{}_{}.png".format(fn.split('.')[0],idx), aug_mask)
        print("../data/validation/aug_mask/{}_{}.png".format(fn.split('.')[0],idx))



