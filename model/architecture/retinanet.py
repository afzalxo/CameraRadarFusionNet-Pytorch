import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from model.architecture.vgg import Vggmax

class Retinanet(nn.Module):
    def __init__(self, feature_size, num_values_regression, num_anchors, num_classes, image_size=(360, 640)):
        super(Retinanet, self).__init__()
        self.feature_size = feature_size
        self.num_values_regression = num_values_regression
        self.num_anchors = num_anchors
        self.pyramid_feature_size = 256
        self.regression_feature_size = 256
        self.classification_feature_size = 256
        self.num_classes = num_classes

        self.backbone = Vggmax()
        __feature_size = self.backbone._feature_sizes()
        self.p5_conv1 = nn.Conv2d(in_channels=__feature_size[-1], out_channels=self.feature_size, kernel_size=1, stride=1, padding=0)
        self.p5_conv2 = nn.Conv2d(in_channels=self.feature_size, out_channels=self.feature_size, kernel_size=3, stride=1, padding=1)
        self.p5_upsample = transforms.Resize((int(image_size[0]/16), int(image_size[1]/16)), interpolation=InterpolationMode.NEAREST)

        self.p4_conv1 = nn.Conv2d(in_channels=__feature_size[-2], out_channels=self.feature_size, kernel_size=1, stride=1, padding=0)
        self.p4_conv2 = nn.Conv2d(in_channels=self.feature_size, out_channels=self.feature_size, kernel_size=3, stride=1, padding=1)
        self.p4_upsample = transforms.Resize((int(image_size[0]/8), int(image_size[1]/8)), interpolation=InterpolationMode.NEAREST)

        self.p3_conv1 = nn.Conv2d(in_channels=__feature_size[-3], out_channels=self.feature_size, kernel_size=1, stride=1, padding=0)
        self.p3_conv2 = nn.Conv2d(in_channels=self.feature_size, out_channels=self.feature_size, kernel_size=3, stride=1, padding=1)

        self.p6_conv = nn.Conv2d(in_channels=__feature_size[-1], out_channels=self.feature_size, kernel_size=3, stride=2, padding=1)
        self.p7_conv = nn.Conv2d(in_channels=self.feature_size, out_channels=self.feature_size, kernel_size=3, stride=2, padding=1)

        ### Regression ops here
        self.regression_ops = []
        inp_channels = self.pyramid_feature_size
        for i in range(4):
            self.regression_ops += [nn.Conv2d(in_channels=inp_channels, out_channels=self.regression_feature_size, kernel_size=3, stride=1, padding=1)]  #TODO: Kernel initializer to normal pending
            inp_channels = self.regression_feature_size
            self.regression_ops += [nn.ReLU(inplace=False)]
        self.regression_ops += [nn.Conv2d(in_channels=self.regression_feature_size, out_channels=self.num_anchors*self.num_values_regression, kernel_size=3, stride=1, padding=1)] #TODO: Kernel initializer to normal pending

        ### Classification ops here
        self.classification_ops = []
        for i in range(4):
            self.classification_ops += [nn.Conv2d(in_channels=inp_channels, out_channels=self.classification_feature_size, kernel_size=3, stride=1, padding=1)]
            inp_channels = self.classification_feature_size
            self.classification_ops += [nn.ReLU(inplace=False)]
        self.classification_ops += [nn.Conv2d(in_channels=self.classification_feature_size, out_channels=self.num_classes*self.num_anchors, kernel_size=3, stride=1, padding=1)]

    def create_pyramid_features(self, concat_features, radar_layers=None):
        p5 = self.p5_conv1(concat_features[-1])
        p5_upsampled = self.p5_upsample(p5)
        p5 = self.p5_conv2(p5)

        p4 = self.p4_conv1(concat_features[-2])
        print(p4.size(), p5_upsampled.size())
        p4 += p5_upsampled
        p4_upsampled = self.p4_upsample(p4)
        p4 = self.p4_conv2(p4)

        p3 = self.p3_conv1(concat_features[-3])
        p3 += p4_upsampled
        p3 = self.p3_conv2(p3)

        p6 = self.p6_conv(concat_features[-1])

        p7 = nn.ReLU(inplace=False)(p6)
        p7 = self.p7_conv(p7)

        r3 = radar_layers[0]
        r4 = radar_layers[1]
        r5 = radar_layers[2]
        r6 = radar_layers[3]
        r7 = radar_layers[4]

        p3 = torch.cat((p3, r3), axis=1) 
        p4 = torch.cat((p4, r4), axis=1) 
        p5 = torch.cat((p5, r5), axis=1) 
        print(p3.size(), p4.size(), p5.size(), p6.size(), p7.size())
        print(r3.size(), r4.size(), r5.size(), r6.size(), r7.size())
        p6 = torch.cat((p6, r6), axis=1) 
        p7 = torch.cat((p7, r7), axis=1) 

        return [p3, p4, p5, p6, p7]

    def run_regression_submodel(self, features, num_values):
        for i in range(len(self.regression_ops)):
            features = self.regression_ops[i](features)
        features = torch.permute(features, (0, 2, 3, 1))
        outputs = torch.reshape(features, (-1, num_values))
        return outputs

    def run_classification_submodel(self, features, num_classes):
        for i in range(len(self.classification_ops)):
            features = self.classification_ops[i](features)
        features = torch.permute(features, (0, 2, 3, 1))
        outputs = torch.reshape(features, (-1, num_classes))
        outputs = nn.Sigmoid()(outputs)
        return outputs

    def forward(self, input):
        image_features, radar_features = self.backbone(input)
        pyramid_features = self.create_pyramid_features(concat_features=image_features, radar_layers=radar_features) 
        print('--=='*20)
        for feature in pyramid_features:
            #print(feature.size())
            res = self.run_regression_submodel(feature, 4)
            print(res.size())
        regression_out = torch.cat([self.run_regression_submodel(feature, 4) for feature in pyramid_features], dim=0)
        classification_out = torch.cat([self.run_classification_submodel(feature, self.num_classes) for feature in pyramid_features], dim=0)
        return [regression_out, classification_out]
        
