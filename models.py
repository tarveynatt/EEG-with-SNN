import torch
from torch import nn


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


class VGG(nn.Module):
    def __init__(self, vgg='VGG16', cq=False, T=32, clamp_min=0, clamp_max=1, category=10, in_channels=5):
        super(VGG, self).__init__()
        self.cq = cq
        self.T = T
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.category = category
        self.in_channels = in_channels
        self.features = self._make_features(cfg[vgg])
        self.avepool = nn.AdaptiveAvgPool2d(7)
        self.classifier = self._make_classifier()
        
    
    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        out = self.features(x)
        out = self.avepool(out)
        out = self.classifier(out)
        return out
    
    def _make_features(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [
                    nn.AvgPool2d(kernel_size=2, stride=2), 
                    ]
            else:
                if self.cq:
                    layers += [
                    nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    # nn.Dropout2d(0.5)
                    Clamp(max=self.clamp_max, min=self.clamp_min),
                    Quantize(T=self.T)
                    ]
                else:
                    layers += [
                        nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True),
                        # nn.Dropout2d(0.5)
                        ]

                self.in_channels = x

        return nn.Sequential(*layers)

    def _make_classifier(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000),
            nn.Linear(1000, self.category)
        )

