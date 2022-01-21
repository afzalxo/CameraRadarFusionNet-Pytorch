import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import argparse
import numpy as np
import collections
import gc
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from tensorflow import keras
import tensorflow as tf

from model.architecture.retinanet import Retinanet
from model.architecture.vgg import Vggmax
from utils.config import get_config
from data_processing.generator.crf_main_generator import create_generators
#from model.losses import FocalLoss
#from model.losses_keras import smooth_l1
from model import losses_keras
from model.losses_torch import smooth_l1
from model.losses_torch import focal
from model.anchors import Anchors

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config/default.cfg')
parser.add_argument('--num_epochs', type=int, default=20)
args = parser.parse_args()

def main():
    _anchors = Anchors()

    image_backbone = Vggmax()

    cfg = get_config(args.config)
    train_generator, validation_generator, test_generator, test_night_generator, test_rain_generator = create_generators(cfg, image_backbone)

    num_classes = train_generator.num_classes()
    print('Num Classes: {}'.format(num_classes))

    model = Retinanet(image_backbone, _anchors.num_anchors(), num_classes)
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    print('Num training images: {}'.format(len(train_generator)))
    #sample_inputs, _ = train_generator[0]
    #b = torch.permute(torch.tensor(sample_inputs), (0, 3, 1, 2))
    #anchs = _anchors(b)

    #loss_func = FocalLoss()

    def getfree(sr):
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = (r-a)/(1024**2)  # free inside reserved
        print((sr + ' free: {}').format(f))

    for epoch in range(args.num_epochs):

        epoch_loss = []

        for i in range(len(train_generator)):
            #try:
            img, targets = train_generator[i]
            optimizer.zero_grad()

            img = torch.permute(torch.tensor(img).cuda().float(), (0, 3, 1, 2))
            targets_reg = torch.tensor(targets[0]).cuda().float()
            #print('Img size: {}, Targets size: {}'.format(img.size(), targets_reg.size()))
            model.training = False
            reg_out, clas_out = model(img)

            targets_reg = torch.tensor(targets[0])
            pred_reg = torch.tensor(reg_out.detach().cpu().numpy())
            targets_clas = torch.tensor(targets[1])
            pred_clas = torch.tensor(clas_out.detach().cpu().numpy())
            
            clas_loss = focal()(targets_clas, pred_clas)
            reg_loss = smooth_l1()(targets_reg, pred_reg)
            #print('=Torch= Classification Loss: {}, Regression Loss: {}, Total: {}'.format(clas_loss.mean(), reg_loss.mean(), clas_loss.mean()+reg_loss.mean()))

            '''
            cls_bk, reg_bk = clas_loss, reg_loss
            targets_reg = tf.convert_to_tensor(targets[0])
            pred_reg = tf.convert_to_tensor(reg_out.detach().cpu().numpy())
            targets_clas = tf.convert_to_tensor(targets[1])
            pred_clas = tf.convert_to_tensor(clas_out.detach().cpu().numpy())
            clas_loss = losses_keras.focal()(targets_clas, pred_clas)
            reg_loss = losses_keras.smooth_l1()(targets_reg, pred_reg)
            with tf.Session() as sess:
                #print('--=='*20)
                clas_loss = clas_loss.eval()
                reg_loss = reg_loss.eval()
            #    print('###'*20)
            #    print(reg_loss.eval())
            print('=TF= Classification Loss: {}, Regression Loss: {}, Total: {}'.format(clas_loss.mean(), reg_loss.mean(), clas_loss.mean()+reg_loss.mean()))
            clas_loss, reg_loss = cls_bk, reg_bk
            '''
            #exit(0)
            clas_loss = clas_loss.mean()
            reg_loss = reg_loss.mean()
            loss = clas_loss + reg_loss
            loss = Variable(loss, requires_grad = True)
            if bool(loss == 0):
                continue

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            optimizer.step()

            loss_hist.append(float(loss.detach()))
            epoch_loss.append(float(loss.detach()))

            print('Epoch: {} | Iteration: {} | Classification Loss: {:1.5f} | Regression Loss: {:1.5f} | Running Loss: {:1.5f}'.format(epoch, i, float(clas_loss), float(reg_loss), np.mean(loss_hist)))

            del img, targets, clas_loss, reg_loss, loss, targets_reg
            #gc.collect()

            #except Exception as e:
            #    print(e)
            #    continue

        scheduler.step(np.mean(epoch_loss))

if __name__ == '__main__':
    main()


'''
inputs, targets = train_generator[0]
inputs = torch.tensor(inputs).cuda()
print('Input Shapes: {}'.format(inputs.shape))
print('Target Shapes: Regression {}, Classification {}'.format(targets[0].shape, targets[1].shape))

#b = torch.tensor(inputs, dtype=torch.float).cuda()
b = torch.permute(inputs, (0, 3, 1, 2))

res = model(b)
anchs = _anchors(b)
print('Anchor Size: {}'.format(anchs.size()))

classification_loss, regression_loss = FocalLoss()(res[1], res[0], anchs, targets[0])

print('Res[0] size {} and Res[1] size {}'.format(res[0].size(), res[1].size()))
'''
