import torch
from torch import nn
import torch.nn.functional as F


def transfer_cq(src, dst):
    # src: CNN 
    # dst: CNN_CQ
    src_dict = src.state_dict()
    dst_dict = dst.state_dict()

    transfer_dict = {}
    
    for s, d in zip(src_dict.keys(), dst_dict.keys()):
        transfer_dict[d] = nn.Parameter(src_dict[s].float())
    
    dst.load_state_dict(transfer_dict, strict=False)