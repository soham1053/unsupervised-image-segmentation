import torch.nn as nn
import torch.nn.functional as F


class UISNet(nn.Module):
    ''' Unsupervised Image Segmentation Net '''
    
    def __init__(self, nPixelFeatures, nMidConvs):
        super(UISNet, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(3, nPixelFeatures, kernel_size=3, padding=1))
        for _ in range(nMidConvs):
            self.convs.append(nn.Conv2d(nPixelFeatures, nPixelFeatures, kernel_size=3, padding=1))
        self.convs.append(nn.Conv2d(nPixelFeatures, nPixelFeatures, kernel_size=1))
        
        self.bns = nn.ModuleList()
        for _ in range(nMidConvs+2):
            self.bns.append(nn.BatchNorm2d(nPixelFeatures))

    def forward(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = F.relu(x)
            x = self.bns[i](x)
        return x
