import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.bn_dw = nn.BatchNorm2d(nin, eps=1e-5)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        self.bn_pw = nn.BatchNorm2d(nout, eps=1e-5)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = F.relu(x, inplace=True) 
        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = F.relu(x, inplace=True)
        return x

class Inception(nn.Module):

  def __init__(self):
    super(Inception, self).__init__()
    self.branch1x1 = BasicConv2d(128, 43, kernel_size=1, padding=0)
    self.branch1x1_2 = BasicConv2d(128, 43, kernel_size=1, padding=0)
    self.branch3x3_reduce = BasicConv2d(128, 24, kernel_size=1, padding=0)
    self.branch3x3 = BasicConv2d(24, 42, kernel_size=3, padding=1)

  
  def forward(self, x):
    branch1x1 = self.branch1x1(x)
    
    branch1x1_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch1x1_2 = self.branch1x1_2(branch1x1_pool)
    
    branch3x3_reduce = self.branch3x3_reduce(x)
    branch3x3 = self.branch3x3(branch3x3_reduce)

    outputs = [branch1x1, branch1x1_2, branch3x3]
    return torch.cat(outputs, 1)


class CRelu(nn.Module):

  def __init__(self, in_channels, out_channels, **kwargs):
    super(CRelu, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
    self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
  
  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = torch.cat([x, -x], 1)
    x = F.relu(x, inplace=True)
    return x
    
    
class Face(nn.Module):

  def __init__(self, phase, size, num_classes):
    super(Face, self).__init__()
    self.phase = phase
    self.num_classes = num_classes
    self.size = size
    
    self.conv1_1 = BasicConv2d(3, 16, kernel_size=5, stride=4, padding=1)
    self.conv1_2 = BasicConv2d(16, 32, kernel_size=3, stride=2, padding=1)
    self.conv2_1 = BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)
    self.conv2_2 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)
    
    self.inception1 = Inception()
    self.inception2 = Inception()
    self.inception3 = Inception()
    self.inception4 = Inception()
    
    self.conv_dw1 = depthwise_separable_conv(128, 128)
    self.conv3_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
   
    self.conv_dw2 = depthwise_separable_conv(256, 128)
    self.conv4_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
    
    self.loc, self.conf = self.multibox(self.num_classes)
    
    if self.phase == 'test':
        self.softmax = nn.Softmax(dim=-1)

    if self.phase == 'train':
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

  def multibox(self, num_classes):
    loc_layers = []
    conf_layers = []
    loc_layers += [nn.Conv2d(128, 18 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(128, 18 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
    return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)
    
  def forward(self, x):
  
    sources = list()
    loc = list()
    conf = list()
    detection_dimension = list()

    x = self.conv1_1(x)
    x = self.conv1_2(x)
    x = self.conv2_1(x)
    x = self.conv2_2(x)
    x = self.inception1(x)
    x = self.inception2(x)
    x = self.inception3(x)
    x = self.inception4(x)
    detection_dimension.append(x.shape[2:])
    sources.append(x)
    x = self.conv_dw1(x)
    x = self.conv3_2(x)
    detection_dimension.append(x.shape[2:])
    sources.append(x)
    x = self.conv_dw2(x)
    x = self.conv4_2(x)
    detection_dimension.append(x.shape[2:])
    sources.append(x)
    
    detection_dimension = torch.tensor(detection_dimension, device=x.device)

    for (x, l, c) in zip(sources, self.loc, self.conf):
        loc.append(l(x).permute(0, 2, 3, 1).contiguous())
        conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
    loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
    conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

    if self.phase == "test":
      output = (loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(-1, self.num_classes)),
                detection_dimension)
    else:
      output = (loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                detection_dimension)
  
    return output
