import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict

OPS = {
  'none' : lambda C, stride, affine, edge: Zero(stride),
  'noise': lambda C, stride, affine, edge: NoiseOp(stride, 0., 1.),
  'skip_connect' : lambda C, stride, affine, edge: Identity() if stride == 1 else FactorizedReduce(C, C, edge, affine=affine),
  'max_pool_3x3' : lambda C, stride, affine, edge: layers.MaxPooling2D((3, 3), strides=stride, padding='SAME'),
  'avg_pool_3x3' : lambda C, stride, affine, edge: layers.AvgPooling2D((3, 3), strides=stride, padding='SAME'),
  'sep_conv_3x3' : lambda C, stride, affine, edge: SepConv(C, C, 3, stride, 1, edge, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine, edge: SepConv(C, C, 5, stride, 2, edge, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine, edge: SepConv(C, C, 7, stride, 3, edge, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine, edge: DilConv(C, C, 3, stride, 2, 2, edge, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine, edge: DilConv(C, C, 5, stride, 4, 2, edge, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine, edge: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
}

class NoiseOp(nn.Module):
    def __init__(self, stride, mean, std):
        super(NoiseOp, self).__init__()
        self.stride = stride
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.stride != 1:
            x_new = x[:,:,::self.stride,::self.stride]
        else:
            x_new = x
        noise = Variable(x_new.data.new(x_new.size()).normal_(self.mean, self.std))

        return noise


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, edge, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(OrderedDict([('relu',
      nn.ReLU(inplace=False)),('reluconvbnk'+str(kernel_size)+'_convkxk_edge-'+str(edge),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)), ('reluconvbnk'+str(kernel_size)+'_bn_edge-'+str(edge),
      nn.BatchNorm2d(C_out, affine=affine))])
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, edge, affine=True):
    super(DilConv, self).__init__()
    
    self.op = nn.Sequential(OrderedDict([('relu',
      nn.ReLU(inplace=False)), ('dilconvk'+str(kernel_size)+'_convkxk_edge-'+str(edge),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False)), ('dilconvk'+str(kernel_size)+'_conv1x1_edge-'+str(edge),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)), ('dilconvk'+str(kernel_size)+'_bn_edge-'+str(edge),
      nn.BatchNorm2d(C_out, affine=affine))])
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, edge, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(OrderedDict([('relu0',
      nn.ReLU(inplace=False)),('sepconvk'+str(kernel_size) + '_convkxk-0_edge-' + str(edge),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False)), ('sepconvk'+str(kernel_size) + '_conv1x1-0_edge-' +str(edge),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False)), ('sepconvk'+str(kernel_size) + '_bn-0_edge-' +str(edge),
      nn.BatchNorm2d(C_in, affine=affine)), ('relu1',
      nn.ReLU(inplace=False)), ('sepconvk'+str(kernel_size) + '_convkxk-1_edge-' +str(edge),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False)), ('sepconvk'+str(kernel_size) + '_conv1x1-1_edge-' +str(edge),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)), ('sepconvk'+str(kernel_size) + '_bn-1_edge-' +str(edge),
      nn.BatchNorm2d(C_out, affine=affine))])
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

'''
class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)
'''

class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride
  def forward(self, x):
    n, c, h, w = x.size()
    h //= self.stride
    w //= self.stride
    if x.is_cuda:
      with torch.cuda.device(x.get_device()):
        padding = torch.cuda.FloatTensor(n, c, h, w).fill_(0)
    else:
      padding = torch.FloatTensor(n, c, h, w).fill_(0)
    return padding    

class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, edge, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.Sequential(OrderedDict([('skip_relu_' + str(edge), nn.ReLU(inplace=False))]))
    self.conv_1 = nn.Sequential(OrderedDict([('skip_conv1_'+str(edge), nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False))]))
    self.conv_2 = nn.Sequential(OrderedDict([('skip_conv2_'+str(edge), nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False))])) 
    self.bn = nn.Sequential(OrderedDict([('skip_bn_'+str(edge), nn.BatchNorm2d(C_out, affine=affine))])) 

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

