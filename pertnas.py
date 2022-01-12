import torch
import torch.nn as nn
from model.architectures.operations import *
from torch.autograd import Variable
from model.architectures.utils import drop_path

from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

DARTS_CIFAR10 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 2), ('avg_pool_3x3', 4)], reduce_concat=range(2, 6))

class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction=False, reduction_prev=False, edges_per_node=[2,2,2,2]):
        super(Cell, self).__init__()
        self._edges_per_node = edges_per_node
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, edge=0)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, edge=0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, edge=0)    
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        #print(op_names, indices)
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True, edge=0)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob=0.3):
        assert(self._steps == len(self._edges_per_node))
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        ops_list, state_list, out_list = [], [], []
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
            
        return torch.cat([states[i] for i in self._concat], dim=1)


class PertNAS(nn.Module):
    def __init__(self,
            classes=1000,
            cfg=None,
            genotype=DARTS_CIFAR10,
            **kwargs):
        super(PertNAS, self).__init__()

        #fusion_blocks = cfg.fusion_blocks    
        #x = Concatenate(axis=3, name='concat_0')([image_input, radar_input])
        C = 16
        C_curr = 48
        self.stem = nn.Sequential(OrderedDict([('stem_conv',
            nn.Conv2d(5, C_curr, 3, padding=1, bias=False)),('stem_bn',
            nn.BatchNorm2d(C_curr))])
        )
        multiplier = 4

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        # Block 1 - Image
        self.cells = nn.ModuleList()
        self.layer_outputs = []
        self.radar_outputs = []
        self.num_blocks = 5
        red_prev = False
        for i in range(self.num_blocks):
            self.cells.append(Cell(genotype, C_prev_prev, C_prev, C_curr, reduction=True, reduction_prev=red_prev))
            red_prev = True
            C_prev_prev, C_prev = C_prev, multiplier*C_curr+2
            if i is not self.num_blocks-2:
                C_curr *= 2
        #x = nn.MaxPool2d(3, stride=2, padding=1)(x)

        #r = nn.MaxPool2d(2, stride=2)(radar_input)

        #x = torch.cat((x, r), axis=3)

        #return x

    def forward(self, input_tensor):
        #image_input = Lambda(lambda x: x[:, :, :, :3], name='image_channels')(input_tensor)
        #radar_input = Lambda(lambda x: x[:, :, :, 3:], name='radar_channels')(input_tensor)
        print(input_tensor)
        image_input = torch.tensor(input_tensor)
        print(image_input.size())
        #exit(0)
        image_input = input_tensor[:,:3,:,:]
        radar_input = input_tensor[:,3:,:,:]
        print(image_input.size())
        r = radar_input
        x = torch.cat((image_input, radar_input), axis=1)
        x = x_prev = self.stem(x)
        #print(x.size(), x_prev.size())
        x = self.cells[0](x_prev, x)
        for i in range(1, self.num_blocks):
            r = nn.MaxPool2d(2, stride=2)(r)
            x = torch.cat((x, r), axis=1)
            if i > 2:
                self.layer_outputs.append(x)
            self.radar_outputs.append(r)
            x_back = x
            x = self.cells[i](x_prev, x)
            x_prev = x_back 
        r = nn.MaxPool2d(2, stride=2)(r)
        x = torch.cat((x, r), axis=1)
        self.layer_outputs.append(x)
        self.radar_outputs.append(r)
        #Radar outputs for blocks 6 and 7
        r = nn.MaxPool2d(2, stride=2)(r)
        self.radar_outputs.append(r)
        r = nn.MaxPool2d(2, stride=2)(r)
        self.radar_outputs.append(r)
        #print(x.size())
        #x = nn.MaxPool2d()

        #x = self.cell1(x, x)
        #x = nn.MaxPool2d(3, stride=2, padding=1)(x)
        #r = nn.MaxPool2d(2, stride=2)(radar_input)
        #x = torch.cat((x, r), axis=3)
        return self.layer_outputs, self.radar_outputs
    '''
    x = layers.Conv2D(int(64 * cfg.network_width), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(x)
    x = layers.Conv2D(int(64 * cfg.network_width), (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    '''

