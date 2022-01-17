import torch
import torch.nn as nn
from utils.image import preprocess_image
from collections import OrderedDict

class Vggmax(nn.Module):
    def __init__(self):
        super(Vggmax, self).__init__()

        self.block1 = nn.Sequential(OrderedDict([
            ('block1_conv1', nn.Conv2d(in_channels=5, out_channels=64, kernel_size=3, stride=1, padding=1)),
            ('block1_conv1relu', nn.ReLU(inplace=False)),
            ('block1_conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)),
            ('block1_conv2relu', nn.ReLU(inplace=False)),
            ('block1_mp', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        ]))
        self.block2 = nn.Sequential(OrderedDict([
            ('block2_conv1', nn.Conv2d(in_channels=66, out_channels=128, kernel_size=3, stride=1, padding=1)),
            ('block2_conv1relu', nn.ReLU(inplace=False)),
            ('block2_conv2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)),
            ('block2_conv2relu', nn.ReLU(inplace=False)),
            ('block2_mp', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        ]))
        self.block3 = nn.Sequential(OrderedDict([
            ('block3_conv1', nn.Conv2d(in_channels=130, out_channels=256, kernel_size=3, stride=1, padding=1)),
            ('block3_conv1relu', nn.ReLU(inplace=False)),
            ('block3_conv2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)),
            ('block3_conv2relu', nn.ReLU(inplace=False)),
            ('block3_conv3', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)),
            ('block3_conv3relu', nn.ReLU(inplace=False)),
            ('block3_mp', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        ]))
        self.block4 = nn.Sequential(OrderedDict([
            ('block4_conv1', nn.Conv2d(in_channels=258, out_channels=512, kernel_size=3, stride=1, padding=1)),
            ('block4_conv1relu', nn.ReLU(inplace=False)),
            ('block4_conv2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)),
            ('block4_conv2relu', nn.ReLU(inplace=False)),
            ('block4_conv3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)),
            ('block4_conv3relu', nn.ReLU(inplace=False)),
            ('block4_mp', nn.MaxPool2d(kernel_size=2, stride=2, padding=(1,0)))
        ]))
        self.block5 = nn.Sequential(OrderedDict([
            ('block5_conv1', nn.Conv2d(in_channels=514, out_channels=512, kernel_size=3, stride=1, padding=1)),
            ('block5_conv1relu', nn.ReLU(inplace=False)),
            ('block5_conv2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)),
            ('block5_conv2relu', nn.ReLU(inplace=False)),
            ('block5_conv3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)),
            ('block5_conv3relu', nn.ReLU(inplace=False)),
            ('block5_mp', nn.MaxPool2d(kernel_size=2, stride=2, padding=(1,0)))
        ]))

        self.rad_block1_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.rad_block2_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.rad_block3_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.rad_block4_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=(1,0))
        self.rad_block5_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=(1,0))
        self.rad_block6_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.rad_block7_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1)) 

    def _feature_sizes(self):
        return [66, 130, 258, 514, 514]

    def preprocess_image(self, inputs):
        return preprocess_image(inputs, mode='caffe')

    def forward(self, input):
        concat_out = []
        radar_out = []
        image_input = input[:,:3,:,:]
        radar_input = input[:,3:,:,:]
        x = self.block1(input)
        y = self.rad_block1_pool(radar_input)
        x = torch.cat((x, y), axis=1)
        x = self.block2(x)
        y = self.rad_block2_pool(y)
        x = torch.cat((x, y), axis=1)
        x = self.block3(x)
        y = self.rad_block3_pool(y)
        radar_out.append(y)
        x = torch.cat((x, y), axis=1)
        concat_out.append(x)
        x = self.block4(x)
        y = self.rad_block4_pool(y)
        radar_out.append(y)
        x = torch.cat((x, y), axis=1)
        concat_out.append(x)
        x = self.block5(x)
        y = self.rad_block5_pool(y)
        radar_out.append(y)
        x = torch.cat((x, y), axis=1)
        concat_out.append(x)
        x = self.global_avg_pool(x)
        y = self.rad_block6_pool(y) 
        radar_out.append(y)
        y = self.rad_block7_pool(y)
        radar_out.append(y)
        return concat_out, radar_out

