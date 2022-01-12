import argparse
import torch
import numpy as np
from model.architecture.retinanet import Retinanet
from model.architecture.vgg import Vggmax
from utils.config import get_config
from data_processing.generator.crf_main_generator import create_generators

feature_size = 254
num_values_regression = 4
num_anchors = 9

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config/default.cfg')
args = parser.parse_args()
a = Vggmax()
cfg = get_config(args.config)
train_gen, _, _, _, _ = create_generators(cfg, a)

num_classes = train_gen.num_classes()
print('Num Classes:')
print(num_classes)

inputs, targets = train_gen[0]
print('Input Shapes:')
print(inputs.shape)
print('Target Shapes:')
print(targets[0].shape, targets[1].shape)

a = Retinanet(feature_size, num_values_regression, num_anchors, 9)

b = torch.tensor(np.ones((1, 5, 360, 640)), dtype=torch.float)

res = a(b)
print('Res[0] size and Res[1] size')
print(res[0].size(), res[1].size())
