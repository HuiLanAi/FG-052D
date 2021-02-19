#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn
import itertools
import torch.nn.functional as nn_func
import numpy as np

a = torch.zeros((2, 2, 2))
b = np.array([[[1, 1], [2, 2]], [[13, 13], [14, 14]]])
b = torch.from_numpy(b)

print(a)
print(b)

for i in range (2):
    a = a + b[i]
    print(a)
