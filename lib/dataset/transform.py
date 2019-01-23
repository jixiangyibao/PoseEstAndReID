import torch
import torchvision.transforms.functional as F
import random
import numpy as np
from PIL import Image
import cv2

"""We expect a list `cfg.transform_list`. The types specified in this list 
will be applied sequentially. Each type name corresponds to a function name in 
this file, so you have to implement the function w.r.t. your custom type. 
The function head should be `FUNC_NAME(in_dict, cfg)`, and it should modify `in_dict`
in place.
The transform list allows us to apply optional transforms in any order, while custom
functions allow us to perform sync transformation for images and all labels.
"""


def hflip(in_dict, cfg):
    # Tricky!! random.random() can not reproduce the score of np.random.random(),
    # dropping ~1% for both Market1501 and Duke GlobalPool.
    # if random.random() < 0.5:
    if np.random.random() < 0.5:
        in_dict['im'] = F.hflip(in_dict['im'])

# Resize image using cv2.resize()
def resize(in_dict, cfg):
    in_dict['im'] = Image.fromarray(cv2.resize(np.array(in_dict['im']), tuple(cfg.RESIZED_IMAGE_SIZE[::-1]), interpolation=cv2.INTER_LINEAR))

def to_tensor(in_dict, cfg):
    in_dict['im'] = F.to_tensor(in_dict['im'])
    in_dict['im'] = F.normalize(in_dict['im'], cfg.IMG_MEAN, cfg.IMG_STD)

def transform(in_dict, cfg):
    for t in cfg.TRANSFORM_LIST:
        eval('{}(in_dict, cfg)'.format(t))
    to_tensor(in_dict, cfg)
    return in_dict
