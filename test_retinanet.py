import argparse
import torch
import numpy as np
from model.architecture.retinanet import Retinanet
from model.architecture.vgg import Vggmax
from utils.config import get_config
from data_processing.generator.crf_main_generator import create_generators
from model import losses
from model import anchors

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config/default.cfg')
args = parser.parse_args()

num_anchors = 9

image_backbone = Vggmax()
cfg = get_config(args.config)
train_generator, validation_generator, test_generator, test_night_generator, test_rain_generator = create_generators(cfg, image_backbone)

num_classes = train_generator.num_classes()
print('Num Classes: {}'.format(num_classes))

inputs, targets = train_generator[0]
print('Input Shapes: {}'.format(inputs.shape))
print('Target Shapes: Regression {}, Classification {}'.format(targets[0].shape, targets[1].shape))

b = torch.tensor(inputs, dtype=torch.float)
b = torch.permute(b, (0, 3, 1, 2))

model = Retinanet(image_backbone, num_anchors, num_classes)
res = model(b)
anch = anchors.Anchors()(b)
print('Anchor Size: {}'.format(anch.size()))

los = losses.FocalLoss()(res[1], res[0], anch, targets[0])

print('Res[0] size {} and Res[1] size {}'.format(res[0].size(), res[1].size()))
#print(los)
