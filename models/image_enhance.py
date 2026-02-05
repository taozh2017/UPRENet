import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import cv2
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset
import SimpleITK as sitk
from torchvision.utils import save_image
import torchvision.transforms as tr
import dataloaders.image_transforms as myit
import torchvision.transforms as deftfx
import copy

def Support_Enhance(img, mask):
    """
        img enhance
        img:expected 1,1,1,3,256,256
        mask:expected 1,1,1,256,256
    """
    img_t = img[0][0]
    mask_t = mask[0][0]
    # save_image(img_t, "/opt/data/private/TestNet/runs/image/support_img.png")
    # save_image(mask_t,"/opt/data/private/TestNet/runs/image/support_mask.png")
    img_e, mask_e = Geometric(img_t, mask_t)
    
    mask_e = torch.round(mask_e)
    # save_image(img_e, "/opt/data/private/TestNet/runs/image/support_image_enhance.png")
    # save_image(mask_e, "/opt/data/private/TestNet/runs/image/support_mask_enhance.png")

    return [[img_e]], [[mask_e]]

def Query_Enhance(img, mask):
    """
        img enhance
        img:expected 1,1,1,3,256,256
        mask:expected 1,1,1,256,256
    """
    img_t = img[0][0]
    mask_t = mask[0][0]
    # save_image(img_t, "/opt/data/private/TestNet/runs/image/support_img.png")
    # save_image(mask_t,"/opt/data/private/TestNet/runs/image/support_mask.png")
    
    img_e, mask_e = Geometric(img_t, mask_t)

    mask_e = torch.round(mask_e)
    return [[img_e]], [[mask_e]]

def Geometric(img, mask):
    
    mask = mask.unsqueeze(0)
    comp = torch.cat((img, mask), dim=1)
    
    
    transform = tr.Compose([
        tr.RandomPerspective(distortion_scale=0.2, p=1, fill=0),
        tr.RandomAffine(degrees=(-5,5), translate=(0.05,0.05), scale=(0.9,1.1), shear=(-5,5),fill=0),
        tr.ElasticTransform(alpha=(0.5,2.0),sigma=(5.0,10.0),fill=0),
        # tr.RandomHorizontalFlip(p=0.5)
    ])
    
    comp2 = transform(comp)
    
    img_t = comp2[:,:3,:,:]
    mask_t = comp2[:,3:,:,:].squeeze(0)
    
    return img_t, mask_t
