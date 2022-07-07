import torch
import numpy as np
from PIL import Image
from torch.nn import functional as f
from torchvision.transforms import ToTensor, ToPILImage
import random

# tomado de: https://stackoverflow.com/questions/56235733/is-there-a-tensor-operation-or-function-in-pytorch-that-works-like-cv2-dilate
def erode(img,kernel_size):
    # recibe imgs que es torch tensor (B,C,H,W)
    res = img.size()
    kernel_tensor = torch.Tensor(np.ones((1,1,kernel_size,kernel_size)))
    im_tensor = img.unsqueeze(0)
    torch_result = torch.clamp(torch.nn.functional.conv2d(im_tensor, kernel_tensor, padding=(1, 1)), 0, 1)
    post_size = (torch_result.size()[0],torch_result.size()[1])
    if (post_size!=(res[1],res[2])):
        torch_result = torch.nn.functional.interpolate(torch_result,size=(res[1],res[2]), mode='bilinear',align_corners=True)
    return ToPILImage()(torch_result[0][0])


def dilate(img,kernel_size):
    # recibe imgs que es torch tensor (B,C=1,H,W)
    res = img.size()
    kernel_tensor = torch.Tensor(np.ones((1,1,kernel_size,kernel_size)))
    im_tensor = (1-img).unsqueeze(0)
    torch_result = torch.clamp(torch.nn.functional.conv2d(im_tensor, kernel_tensor, padding=(1, 1)), 0, 1)
    post_size = (torch_result.size()[0],torch_result.size()[1])
    if (post_size!=(res[1],res[2])):
        torch_result = torch.nn.functional.interpolate(torch_result,size=(res[1],res[2]), mode='bilinear',align_corners=True)
    return ToPILImage()(1-torch_result[0][0])

def random_morphOp(img, kernel_erode_sizes=(3,),kernel_dilate_sizes=(3,5)):
    op = random.randint(0,1)
    if (op):
        kernel_size = random.choice(kernel_erode_sizes)
        return erode(img, kernel_size)
    elif(op==0):
        kernel_size = random.choice(kernel_dilate_sizes)
        return dilate(img, kernel_size)
    else:
        return ToPILImage()(img)

    