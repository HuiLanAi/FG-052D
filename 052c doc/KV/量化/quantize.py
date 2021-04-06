import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import os
import torch.nn.utils.prune as torch_prune
import copy


# quantize function
# input: old_tensor - original data format's tensor
# input: inte_w - the width of integer part
# input: deci_w - the width of decimal part
# note: sign bit defaultly occupies one bit
# output: q_tensor - the quantized version of tensor 
def quantize(old_tensor, inte_w, deci_w):
    shift = 2 ** (int)(deci_w)
    upper = 2 ** (int)(deci_w + inte_w)
    with torch.no_grad():
        old_tensor *= shift
        old_tensor = old_tensor.int()
        old_tensor[old_tensor > upper] = upper
        old_tensor[old_tensor < -upper] = -upper
        old_tensor = old_tensor.float()
        old_tensor /= shift
    return old_tensor



old_tensor = torch.tensor([[0, 0, 0], [0, 0, 0]])
old_tensor = quantize(old_tensor, 3, 8)
print(old_tensor)