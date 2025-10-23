import numpy as np
import os
import random
import torch

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    #os.environ['PYTHONHASHSEED'] = str(seed)
    #torch.backends.cudnn.determinisitc = True
