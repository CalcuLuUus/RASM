import torch
import torch.nn as nn
import torch.nn.functional as F


def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
    
from torchvision import models

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.vgg_pretrained_features = models.vgg19(pretrained=True).features
    def forward(self, X, indices=None):
        if indices is None:
            indices = [2, 7, 12, 21, 30]
        out = []
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            if (i + 1) in indices:
                out.append(X)
        return out

class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class VGGLossV2(nn.Module):
    def __init__(self, vgg=None, weights=None):
        super(VGGLossV2, self).__init__()
        #sself.vgg = Vgg19().cuda()
        self.vgg = Vgg19().cuda()
        #torch.compile(Vgg19().cuda()) # Vgg19().cuda() 
        self.criterion = nn.L1Loss()
        self.weights = weights or [0.1, 0.1, 1, 1, 1]
        #[1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        self.indices = [2, 7, 12, 21, 30]
    def forward(self, x, y):

        x, y = self.normalize(x), self.normalize(y)

        x_vgg = self.vgg(x, self.indices)
        with torch.no_grad():
            y_vgg = self.vgg(y, self.indices)
        # x_vgg, y_vgg = self.vgg(torch.cat([x, y])).chunk(2)
        loss = 0
        for w, fx, fy in zip(self.weights, x_vgg, y_vgg):
            loss += w * self.criterion(fx, fy)
        return loss

class L1_Vgg_losses(nn.Module):
    def __init__(self):
        super(L1_Vgg_losses, self).__init__()
        self.l1 = CharbonnierLoss()
        self.vgg = VGGLossV2()
    def forward(self, x, y):
        return self.l1(x, y) + 0.001 * self.vgg(x, y)

class CrossEntropyLoss(nn.Module):
    """Cross Entropy Loss"""

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, input, target):
        # print(input.dtype, target.dtype)
        return self.bce(input, target)
    
class Vgg_loss(nn.Module):
    def __init__(self):
        super(Vgg_loss, self).__init__()
        self.vgg = VGGLossV2()
    def forward(self, x, y):
        return 0.001 * self.vgg(x, y)