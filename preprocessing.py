import cv2 as cv
import os
# MASK_PATH = "../data/masks/"
# IMG_PATH = "../data/images/"
# for fn in os.listdir(MASK_PATH):
#     img = cv.imread(os.path.join(IMG_PATH,fn.split('.')[0]+'.jpg'))
#     # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     threshold = cv.inRange(img,(0, 0, 0), (5, 10, 10))
#     mask = cv.imread(os.path.join(MASK_PATH,fn))
#     mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
#     for i in range(mask.shape[0]):
#         for j in range(mask.shape[1]):
#             if threshold[i][j]==255:
#                 mask[i][j] = 0
#     # mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
#     cv.imwrite(os.path.join(MASK_PATH,fn),mask)
#     print("write image {}".format(fn))

MASK_PATH="../data/validation/masks/"
IMG_PATH = "../data/validation/images/"

for fn in os.listdir(MASK_PATH):
    img = cv.imread(os.path.join(IMG_PATH,fn.split('.')[0]+'.jpg'))
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    threshold = cv.inRange(img,(0, 0, 0), (5, 10, 10))
    mask = cv.imread(os.path.join(MASK_PATH,fn))
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if threshold[i][j]==255:
                mask[i][j] = 0
    # mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    cv.imwrite(os.path.join(MASK_PATH,fn),mask)
    print("write image {}".format(fn))