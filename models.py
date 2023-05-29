import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class RevGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None

revgrad = RevGrad.apply

class RevGrad(nn.Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.

        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)
        self._alpha = torch.tensor(alpha, requires_grad=False)
    def forward(self, input_):
        return revgrad(input_, self._alpha)
def grad_reverse(x, lambd=1.0):
    return RevGrad(lambd)(x)


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.num_class = args.known_class + 1
        self.generator = ResBase()
        self.classifier = Classifier(self.num_class, unit_size=1024)
 
        dim = 1024 if args.all_layer_adv else 2048
        self.adv_k = AdversarialNetwork(dim)
        self.adv_unk = AdversarialNetwork(dim)

    def forward(self, x, constant=1, adaption=False):
        rois = self.generator(x)
        x = self.classifier(rois, constant, adaption)
        return  x

class ResBase(nn.Module):
    def __init__(self):
        super(ResBase, self).__init__()
        self.CNN = models.resnet50(pretrained=True)
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        modules = list(self.CNN.children())[:-2]
        self.CNN = nn.Sequential(*modules)
    def forward(self, imgs):
        batch_size, _, _, _ = imgs.shape
        rois = self.CNN(imgs)
        return rois

class Classifier(nn.Module):
    def __init__(self, num_classes, unit_size=100):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(2048, unit_size)
        self.bn1 = nn.BatchNorm1d(unit_size, affine=True, track_running_stats=True)
        self.linear2 = nn.Linear(unit_size, unit_size)
        self.bn2 = nn.BatchNorm1d(unit_size, affine=True, track_running_stats=True)
        self.classifier = nn.Linear(unit_size, num_classes)
        self.drop = nn.Dropout(p=0.3)
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, rois, constant=1, adaption=False, pooling=True, return_feat=False):

        if pooling:
            rois = self.average_pooling(rois).view(rois.size(0), -1)

        x = self.drop(F.relu(self.bn1(self.linear1(rois))))
        x = self.drop(F.relu(self.bn2(self.linear2(x))))

        # grl or grad scaling
        x_rev = grad_reverse(x, constant) if adaption else x
        logits = self.classifier(x_rev)

        if return_feat:
            return logits, x
        else:
            return logits

class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()

        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.main1 = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True,),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, constant=0.05):
        x = grad_reverse(x, constant)
        for module in self.main1.children():
            x = module(x)
        return x.view(-1)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
