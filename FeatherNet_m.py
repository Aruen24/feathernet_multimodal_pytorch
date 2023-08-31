import torch.nn as nn
import torch.nn.functional as F
import math
from  torch.autograd import Variable
import torch
from torchsummary import summary
from cbam import CBAMBlock

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True)*0.16666667
        return out

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()

    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()
    )

#reference form : https://github.com/moskomule/senet.pytorch
class SELayer1(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer1, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(channel // reduction),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(channel),
            hsigmoid()
        )
    def forward(self, x):
        return x *self.se(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(),
                nn.Linear(channel // reduction, channel),
                #nn.Sigmoid()
                hsigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y




class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, downsample=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.downsample = downsample

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #nn.ReLU6(inplace=True),
                nn.Hardswish(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #nn.ReLU6(inplace=True),
                nn.Hardswish(),

                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #nn.ReLU6(inplace=True),
                nn.Hardswish(),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            if self.downsample is not None:
                return self.downsample(x) + self.conv(x)
            else:
                return self.conv(x)


class FeatherNet(nn.Module):
    def __init__(self, num_class, input_size,input_channels, is_training = True,se=False, cbam=False, avgdown=False, width_mult=1.):
        super(FeatherNet, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1024
        self.se = se
        self.cbam = cbam
        self.avgdown = avgdown
        self.is_training = is_training
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 2],
            [6, 32, 2, 2],  # 56x56
            [6, 48, 6, 2],  # 14x14
            [6, 64, 3, 2],  # 7x7
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(input_channels, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                downsample = None
                if i == 0:
                    if self.avgdown:
                        downsample = nn.Sequential(nn.AvgPool2d(2, stride=2),
                                                   nn.BatchNorm2d(input_channel),
                                                   nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=False)
                                                   )
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, downsample=downsample))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, downsample=downsample))
                input_channel = output_channel
            if self.se:
                self.features.append(SELayer1(input_channel))
            if self.cbam:
                self.features.append(CBAMBlock(input_channel))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        #         building last several layers
        self.final_DW = nn.Sequential(nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=2, padding=1,
                                                groups=input_channel, bias=False))
        #add by lcw

        # traning
        if self.is_training:
            self.avg = nn.AdaptiveAvgPool2d(1)
            self.flatten = Flatten()
            self.logits = nn.Sequential(nn.Dropout(0.5), nn.Linear(64, num_class))
            self.predictions = nn.Softmax(dim=1)
            self._initialize_weights()
        else:
            # test
            self.predictions = nn.Sequential(nn.Linear(64, num_class), nn.Softmax(dim=1))
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.final_DW(x)
        #x = x.mean(3).mean(2)
        x=self.avg(x)
        x=self.flatten(x)


        """training"""
        if self.is_training:
            logits = self.logits(x)
            predictions = self.predictions(logits)
            return logits, predictions
        else:
            """testing"""
            logits = self.logits(x)
            return logits


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                  m.bias.data.zero_()

#Block A and Block C
def FeatherNetA():
    model = FeatherNet(se=True)
    return model

#Block A and Block B
def FeatherNetB():
    #model = FeatherNet(num_class=2,input_size=224,input_channels=1,se=True, avgdown=True)
    model = FeatherNet(num_class=2, input_size=224, input_channels=1, cbam=True, avgdown=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    version=model.to(device)
    summary(version,(1,224,224))

if __name__=='__main__':
     FeatherNetB()
