import torch
import numpy as np
from model.architecture.retinanet import Retinanet
from utils.config import get_config
from data_processing.generator.crf_main_generator import create_generators

feature_size = 254
num_values_regression = 256
num_anchors = 50
num_classes = 50

conf = 'configs/default.cfg'
cfg = get_config(conf)

a = Retinanet(feature_size, num_values_regression, num_anchors, num_classes)
b = torch.tensor(np.ones((1, 5, 360, 640)), dtype=torch.float)

res = a(b)
print(res[0].size(), res[1].size())
