import torch
from torch import nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class Quantize(nn.Module):
    def __init__(self, T):
        super(Quantize, self).__init__()
        self.T = T

    def forward(self, x):
        out = torch.torch.div(torch.floor(torch.mul(x, self.T)), self.T)
        return out


class Clamp(nn.Module):
    def __init__(self, min=0.0, max=1.0):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        out = torch.clamp(x, min=self.min, max=self.max)
        return out


class Flatten(nn.Module):
    def __init__(self, shape):
        super(Flatten, self).__init__()
        self.shape = shape

    def forward(self, x):
        out = x.view(shape, -1)
        return out



class QuantitizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None, nbit=32):
        super(QuantitizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.nbit = nbit
    
    def forward(self, input):
        return F.conv2d(input, self._quantize(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups, )

    def _quantize(self, x):
        return torch.mul(torch.round(torch.div(x, 2.0**(1-self.nbit))), 2.0**(1-self.nbit))


class VGG(nn.Module):
    def __init__(self, vgg='VGG16', cq=False, T=32, clamp_min=0, clamp_max=1, spike=False, category=10, nbit=32):
        super(VGG, self).__init__()
        # self.cq = cq
        # self.T = T
        # self.clamp_min = clamp_min
        # self.clamp_max = clamp_max
        # self.spike = spike
        self.category = category
        # self.nbit = nbit
        self.features = self._make_features(cfg[vgg])
        self.avepool = nn.AdaptiveAvgPool2d(7)
        self.classifier = self._make_classifier()
    
    def forward(self, x):
        out = self.features(x)
        out = self.avepool(out)
        out = self.classifier(out)
        return out
    
    def _make_features(self, cfg):
        layers = []
        in_channels = 3
        spike_grad = surrogate.fast_sigmoid()

        for x in cfg:
            if x == 'M':
                layers += [
                    nn.AvgPool2d(kernel_size=2, stride=2), 
                    # nn.Dropout2d(0.2)
                    ]
            else:
                # if self.cq:
                #     layers += [
                #     nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                #     # QuantitizedConv2d(in_channels, x, kernel_size=3, padding=1, nbit=self.nbit),
                #     nn.BatchNorm2d(x),
                #     Clamp(max=self.clamp_max, min=self.clamp_min),
                #     Quantize(T=self.T),
                #     nn.Dropout2d(0.2)
                #     ]
                # elif self.spike:
                #     layers += [
                #     nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                #     # QuantitizedConv2d(in_channels, x, kernel_size=3, padding=1, nbit=self.nbit),
                #     snn.Leaky(beta=0.5, init_hidden=True, spike_grad=spike_grad),
                #     nn.Dropout2d(0.2),
                #     ]
                # else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                    # nn.Dropout2d(0.2)
                    ]

                in_channels = x

        return nn.Sequential(*layers)

    def _make_classifier(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.Linear(4096, 1000),
            nn.Linear(1000, self.category)
        )

