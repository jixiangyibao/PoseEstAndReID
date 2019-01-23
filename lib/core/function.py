# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import numpy as np
import torch

from lib.core.config import get_model_name
from lib.core.inference import get_max_preds


logger = logging.getLogger(__name__)


def validate(config, val_loader, val_dataset, model):
    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            print(input['im_path'])
            input_im=input['im'].cuda(device=1)
            # compute output
            output = model(input_im)#input：32*3*384*288 output:hratmap 32*17*96*72
            preds, maxvals = get_max_preds(output.clone().cpu().numpy())
            preds=np.squeeze(preds)
            print(preds)
            #print(maxvals)
            preds_original=[[int(x[0]*64/72),int(x[1]*128/96)] for x in preds]

            #preds_original=[[int(coord[0]*config.MODEL.IMAGE_SIZE[0]/config.MODEL.EXTRA.HEATMAP_SIZE[0]), int(coord[1]*config.MODEL.IMAGE_SIZE[1]/config.MODEL.EXTRA.HEATMAP_SIZE[1])]for coord in preds]
            print(preds_original)