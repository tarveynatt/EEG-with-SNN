import torch
from torch import nn
import torch.nn.functional as F


def quantize_to_bit(x, nbit=32):
    return torch.mul(torch.round(torch.div(x, 2.0**(1-nbit))), 2.0**(1-nbit))


def transfer_cq(src, dst):
    # src: CNN 
    # dst: CNN_CQ
    src_dict = src.state_dict()
    dst_dict = dst.state_dict()

    transfer_dict = {}
    
    for s, d in zip(src_dict.keys(), dst_dict.keys()):
        transfer_dict[d] = nn.Parameter(src_dict[s].float())
    
    dst.load_state_dict(transfer_dict, strict=False)


def transfer_snn(src, dst):
    # src: CNN_CQ 
    # dst: SNN
    src_dict = src.state_dict()
    dst_dict = dst.state_dict()
    src_keys = []
    dst_keys = []

    for k in src_dict.keys():
        if 'running_mean' in k:
            src_keys.pop(-1)
            src_keys.pop(-1)
        if 'weight' in k or 'bias' in k:
            src_keys.append(k)

    for k in dst_dict.keys():
        if 'weight' in k or 'bias' in k:
            dst_keys.append(k)
    
    reshape_dict = {}
    for i in range(len(src_keys)):
        reshape_dict[dst_keys[i]] = nn.Parameter(src_dict[src_keys[i]].float())

    dst.load_state_dict(reshape_dict, strict=False)


# fuse data normalization to weight and bias normalizaiton for SNN 
def fuse(model):
    stack = []
    for block in model.named_children():
        if block[0] == 'features': 
            for layer in block[1].children():
                if isinstance(layer, nn.BatchNorm2d):
                    bn_dict = layer.state_dict()
                    conv_dict = stack[-1].state_dict()

                    epsilon = layer.eps
                    mu = bn_dict['running_mean']
                    sigma = bn_dict['running_var']
                    gamma = bn_dict['weight']
                    beta = bn_dict['bias']

                    W = conv_dict['weight']
                    b = conv_dict['bias']

                    coef = gamma.div(torch.sqrt(sigma + epsilon))
                    b.mul_(coef).add_(beta - mu.mul(coef))

                    coef = coef.expand_as(W.transpose(0, -1)).transpose(0, -1)
                    W.mul_(coef)
                    

                    stack[-1].weight.data.copy_(W)
                    stack[-1].bias.data.copy_(b)
                else:
                    stack.append(layer)
    
    model._modules['features'] = nn.Sequential(*stack)
    return model


def normalize_weight(model, threshold_scale=1.0):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            factor = torch.max(torch.abs(m.weight))
            if torch.max(torch.abs(m.bias)) > factor:
                factor = torch.max(torch.abs(m.bias))
            
            m.weight /= factor
            m.weight = nn.Parameter(quantize_to_bit(m.weight))
            m.bias /= factor
            m.bias = nn.Parameter(quantize_to_bit(m.bias))