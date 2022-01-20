import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.ops import nms
from model.architecture.vgg import Vggmax
#from model import losses_torch
from model import losses
from model.anchors import Anchors
from utils.bbox_utils import BBoxTransform, ClipBoxes

class Retinanet(nn.Module):
    def __init__(self, backbone, num_anchors, num_classes, num_values_regression=4, feature_size=254, image_size=(360, 640)):
        super(Retinanet, self).__init__()
        self.feature_size = feature_size
        self.num_values_regression = num_values_regression
        self.num_anchors = num_anchors
        self.pyramid_feature_size = 256
        self.regression_feature_size = 256
        self.classification_feature_size = 256
        self.num_classes = num_classes

        self.backbone = backbone
        __feature_size = self.backbone._feature_sizes()
        self.p5_conv1 = nn.Conv2d(in_channels=__feature_size[-1], out_channels=self.feature_size, kernel_size=1, stride=1, padding=0)
        self.p5_conv2 = nn.Conv2d(in_channels=self.feature_size, out_channels=self.feature_size, kernel_size=3, stride=1, padding=1)
        self.p5_upsample = transforms.Resize((int(image_size[0]/16+1), int(image_size[1]/16)), interpolation=InterpolationMode.NEAREST)
        #self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_conv1 = nn.Conv2d(in_channels=__feature_size[-2], out_channels=self.feature_size, kernel_size=1, stride=1, padding=0)
        self.p4_conv2 = nn.Conv2d(in_channels=self.feature_size, out_channels=self.feature_size, kernel_size=3, stride=1, padding=1)
        self.p4_upsample = transforms.Resize((int(image_size[0]/8), int(image_size[1]/8)), interpolation=InterpolationMode.NEAREST)
        #self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p3_conv1 = nn.Conv2d(in_channels=__feature_size[-3], out_channels=self.feature_size, kernel_size=1, stride=1, padding=0)
        self.p3_conv2 = nn.Conv2d(in_channels=self.feature_size, out_channels=self.feature_size, kernel_size=3, stride=1, padding=1)

        self.p6_conv = nn.Conv2d(in_channels=__feature_size[-1], out_channels=self.feature_size, kernel_size=3, stride=2, padding=1)
        self.p7_conv = nn.Conv2d(in_channels=self.feature_size, out_channels=self.feature_size, kernel_size=3, stride=2, padding=1)

        ### Regression ops here
        self.regression_ops = nn.ModuleList()
        inp_channels = self.pyramid_feature_size
        for i in range(4):
            self.regression_ops += [nn.Conv2d(in_channels=inp_channels, out_channels=self.regression_feature_size, kernel_size=3, stride=1, padding=1)]  #TODO: Kernel initializer to normal pending
            inp_channels = self.regression_feature_size
            self.regression_ops += [nn.ReLU(inplace=False)]
        self.regression_ops += [nn.Conv2d(in_channels=self.regression_feature_size, out_channels=self.num_anchors*self.num_values_regression, kernel_size=3, stride=1, padding=1)] #TODO: Kernel initializer to normal pending

        ### Classification ops here
        self.classification_ops = nn.ModuleList()
        inp_channels = self.pyramid_feature_size
        for i in range(4):
            self.classification_ops += [nn.Conv2d(in_channels=inp_channels, out_channels=self.classification_feature_size, kernel_size=3, stride=1, padding=1)]
            inp_channels = self.classification_feature_size
            self.classification_ops += [nn.ReLU(inplace=False)]
        self.classification_ops += [nn.Conv2d(in_channels=self.classification_feature_size, out_channels=self.num_classes*self.num_anchors, kernel_size=3, stride=1, padding=1)]

        self.out_sig = nn.Sigmoid()

        self.anchors = Anchors()
        #self.focalloss = losses_torch.focal()
        #self.smoothl1 = losses_torch.smooth_l1()
        self.focalloss = losses.FocalLoss()
        self.bboxtransform = BBoxTransform()
        self.clipboxes = ClipBoxes()

    def create_pyramid_features(self, concat_features, radar_layers=None):
        p5 = self.p5_conv1(concat_features[-1])
        p5_upsampled = self.p5_upsample(p5)
        p5 = self.p5_conv2(p5)

        p4 = self.p4_conv1(concat_features[-2])
        #print(p4.shape, p5_upsampled.shape)
        #if p4.shape[2] > p5_upsampled.shape[2]:
        #    p5_upsampled = nn.functional.pad(p5_upsampled, (0, 0, 1, 0))
        #elif p5_upsampled.shape[2] > p4.shape[2]:
        #    p4 = nn.functional.pad(p4, (0, 0, 1, 0))
        #print(p4.shape, p5_upsampled.shape)
        p4 += p5_upsampled
        p4_upsampled = self.p4_upsample(p4)
        p4 = self.p4_conv2(p4)

        p3 = self.p3_conv1(concat_features[-3])
        #print(p3.shape, p4_upsampled.shape)
        #if p3.shape[2] > p4_upsampled.shape[2]:
        #    p4_upsampled = nn.functional.pad(p4_upsampled, (0, 0, 1, 0))
        #elif p4_upsampled.shape[2] > p3.shape[2]:
        #    p3 = nn.functional.pad(p3, (0, 0, 1, 0))
        #print(p3.shape, p4_upsampled.shape)
        p3 += p4_upsampled
        p3 = self.p3_conv2(p3)

        p6 = self.p6_conv(concat_features[-1])

        p7 = nn.ReLU(inplace=False)(p6)
        p7 = self.p7_conv(p7)

        if self.backbone.radar:
            r3 = radar_layers[0]
            r4 = radar_layers[1]
            r5 = radar_layers[2]
            r6 = radar_layers[3]
            r7 = radar_layers[4]

            p3 = torch.cat((p3, r3), axis=1) 
            p4 = torch.cat((p4, r4), axis=1) 
            p5 = torch.cat((p5, r5), axis=1) 
            p6 = torch.cat((p6, r6), axis=1) 
            p7 = torch.cat((p7, r7), axis=1) 

        return [p3, p4, p5, p6, p7]

    def run_regression_submodel(self, features, num_values):
        for i in range(len(self.regression_ops)):
            features = self.regression_ops[i](features)
        features = torch.permute(features, (0, 2, 3, 1))
        outputs = features.contiguous().view(features.shape[0], -1, num_values)
        #outputs = torch.reshape(features, (features.shape[0], -1, num_values))
        #print('Regression Outputs Size:')
        #print(num_values, outputs.size())
        return outputs

    def run_classification_submodel(self, features, num_classes):
        for i in range(len(self.classification_ops)):
            features = self.classification_ops[i](features)
        features = self.out_sig(features)
        features = torch.permute(features, (0, 2, 3, 1))
        batch_size, width, height, channels = features.shape
        features = features.view(batch_size, width, height, self.num_anchors, num_classes)
        outputs = features.contiguous().view(features.shape[0], -1, num_classes)
        #print('Classification Outputs Size:')
        #print(num_classes, outputs.size())
        return outputs

    def forward(self, input):
        if self.training:
            input, annotations = input
        if self.backbone.radar:
            image_features, radar_features = self.backbone(input)
        else:
            image_features, radar_features = self.backbone(input), None
        pyramid_features = self.create_pyramid_features(concat_features=image_features, radar_layers=radar_features) 
        #print('--=='*20)
        #for feature in pyramid_features:
            #print(feature.size())
            #res = self.run_regression_submodel(feature, 4)
            #print(res.size())
        regression_out = torch.cat([self.run_regression_submodel(feature, 4) for feature in pyramid_features], dim=1)
        classification_out = torch.cat([self.run_classification_submodel(feature, self.num_classes) for feature in pyramid_features], dim=1)

        anchors = self.anchors(input)
        
        if self.training:
            fl = self.focalloss(classification_out, regression_out, anchors, annotations)
            return fl
            #sl = self.smoothl1()
            #return fl, sl
            #return self.focalloss(classification_out, regression_out, anchors, annotations)
            
            #rl = self.smoothl1(torch.tensor(annotations[0]).cuda(), regression_out)
            #cl = self.focalloss(torch.tensor(annotations[1]).cuda(), classification_out)
        else:
            transformed_anchors = self.bboxtransform(anchors, regression_out)
            transformed_anchors = self.clipboxes(transformed_anchors, input)

            finalResult = [[], [], []]
            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification_out.shape[2]):
                scores = torch.squeeze(classification_out[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]
        
