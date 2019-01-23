#!/usr/bin/env bash
export PYTHONPATH=~/LY/PeAndReID
/usr/bin/python3 -m pdb lib/pose_estimation/valid.py  --cfg experiments/coco/resnet152/384x288_d256x3_adam_lr1e-3.yaml   --model-file models/pytorch/pose_coco/pose_resnet_152_384x288.pth.tar
