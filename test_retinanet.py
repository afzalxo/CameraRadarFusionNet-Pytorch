import argparse
import numpy as np
import collections
import gc

import torch
import torch.optim as optim
import torch.nn as nn

from model.architecture.retinanet import Retinanet
from model.architecture.vgg import Vggmax
from utils.config import get_config
from data_processing.generator.crf_main_generator import create_generators
from model.losses import FocalLoss
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
    sample_inputs, _ = train_generator[0]
    b = torch.permute(torch.tensor(sample_inputs), (0, 3, 1, 2))
    anchs = _anchors(b)

    loss_func = FocalLoss()

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
            getfree('after train_gen')
            optimizer.zero_grad()

            #img = torch.tensor(img).cuda()
            img = torch.permute(torch.tensor(img).cuda().float(), (0, 3, 1, 2))
            getfree('after permute')
            reg_out, clas_out = model(img)
            getfree('after model')

            #targets_reg = torch.tensor(targets[0]).cuda()
            targets_reg = torch.tensor(targets[0]).cuda().float()
            
            clas_loss, reg_loss = loss_func(clas_out, reg_out, anchs, targets_reg)
            #clas_loss, reg_loss = torch.tensor([0], dtype=float), torch.tensor([0], dtype=float)
            clas_loss = clas_loss.mean()
            reg_loss = reg_loss.mean()
            loss = clas_loss + reg_loss

            if bool(loss == 0):
                continue

            getfree('after loss')
            loss.backward()

            getfree('after firstiter')
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            optimizer.step()

            #loss_hist.append(float(loss.detach()))
            #epoch_loss.append(float(loss.detach()))

            print('Epoch: {} | Iteration: {} | Classification Loss: {:1.5f} | Regression Loss: {:1.5f} | Running Loss: {:1.5f}'.format(epoch, i, float(clas_loss), float(reg_loss), np.mean(loss_hist)))

            del img, targets, reg_out, clas_out, clas_loss, reg_loss, loss, targets_reg
            gc.collect()

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
