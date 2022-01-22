import torch
import torch.nn as nn
from utils.image import preprocess_image
from collections import OrderedDict

class Vggmax(nn.Module):
    def __init__(self, radar):
        super(Vggmax, self).__init__()
        self.radar = radar
        if self.radar:
            self.b1_in_ch = 5
            self.b2_in_ch = 66
            self.b3_in_ch = 130
            self.b4_in_ch = 258
            self.b5_in_ch = 514
        else:
            self.b1_in_ch = 3
            self.b2_in_ch = 64
            self.b3_in_ch = 128
            self.b4_in_ch = 256
            self.b5_in_ch = 512

        self.block1 = nn.Sequential(OrderedDict([
            ('block1_conv1', nn.Conv2d(in_channels=self.b1_in_ch, out_channels=64, kernel_size=3, stride=1, padding=1)),
            ('block1_conv1relu', nn.ReLU(inplace=False)),
            ('block1_conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)),
            ('block1_conv2relu', nn.ReLU(inplace=False)),
            ('block1_mp', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        ]))
        self.block2 = nn.Sequential(OrderedDict([
            ('block2_conv1', nn.Conv2d(in_channels=self.b2_in_ch, out_channels=128, kernel_size=3, stride=1, padding=1)),
            ('block2_conv1relu', nn.ReLU(inplace=False)),
            ('block2_conv2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)),
            ('block2_conv2relu', nn.ReLU(inplace=False)),
            ('block2_mp', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        ]))
        self.block3 = nn.Sequential(OrderedDict([
            ('block3_conv1', nn.Conv2d(in_channels=self.b3_in_ch, out_channels=256, kernel_size=3, stride=1, padding=1)),
            ('block3_conv1relu', nn.ReLU(inplace=False)),
            ('block3_conv2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)),
            ('block3_conv2relu', nn.ReLU(inplace=False)),
            ('block3_conv3', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)),
            ('block3_conv3relu', nn.ReLU(inplace=False)),
            ('block3_mp', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        ]))
        self.block4 = nn.Sequential(OrderedDict([
            ('block4_conv1', nn.Conv2d(in_channels=self.b4_in_ch, out_channels=512, kernel_size=3, stride=1, padding=1)),
            ('block4_conv1relu', nn.ReLU(inplace=False)),
            ('block4_conv2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)),
            ('block4_conv2relu', nn.ReLU(inplace=False)),
            ('block4_conv3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)),
            ('block4_conv3relu', nn.ReLU(inplace=False)),
            ('block4_mp', nn.MaxPool2d(kernel_size=2, stride=2, padding=(1,0)))
        ]))
        self.block5 = nn.Sequential(OrderedDict([
            ('block5_conv1', nn.Conv2d(in_channels=self.b5_in_ch, out_channels=512, kernel_size=3, stride=1, padding=1)),
            ('block5_conv1relu', nn.ReLU(inplace=False)),
            ('block5_conv2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)),
            ('block5_conv2relu', nn.ReLU(inplace=False)),
            ('block5_conv3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)),
            ('block5_conv3relu', nn.ReLU(inplace=False)),
            ('block5_mp', nn.MaxPool2d(kernel_size=2, stride=2, padding=(1,0)))
        ]))

        if radar:
            self.rad_block1_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.rad_block2_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.rad_block3_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.rad_block4_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=(1,0))
            self.rad_block5_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=(1,0))
            self.rad_block6_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.rad_block7_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1)) 

    def _feature_sizes(self):
        #return [66, 130, 258, 514, 514]
        return [self.b2_in_ch, self.b3_in_ch, self.b4_in_ch, self.b5_in_ch, self.b5_in_ch]

    def preprocess_image(self, inputs):
        return preprocess_image(inputs, mode='tf')#mode='caffe')

    def forward(self, input):
        concat_out = []
        if self.radar:
            radar_out = []
            radar_input = input[:,3:,:,:]
            
        x = self.block1(input)
        if self.radar:
            y = self.rad_block1_pool(radar_input)
            x = torch.cat((x, y), axis=1)
        x = self.block2(x)
        if self.radar:
            y = self.rad_block2_pool(y)
            x = torch.cat((x, y), axis=1)
        x = self.block3(x)
        if self.radar:
            y = self.rad_block3_pool(y)
            radar_out.append(y)
            x = torch.cat((x, y), axis=1)
        concat_out.append(x)
        x = self.block4(x)
        if self.radar:
            y = self.rad_block4_pool(y)
            radar_out.append(y)
            x = torch.cat((x, y), axis=1)
        concat_out.append(x)
        x = self.block5(x)
        if self.radar:
            y = self.rad_block5_pool(y)
            radar_out.append(y)
            x = torch.cat((x, y), axis=1)
        concat_out.append(x)
        x = self.global_avg_pool(x)
        if self.radar:
            y = self.rad_block6_pool(y) 
            radar_out.append(y)
            y = self.rad_block7_pool(y)
            radar_out.append(y)
            return concat_out, radar_out
        else:
            return concat_out

