import argparse
import collections
import gc

import numpy as np

import time

import torch
import torch.optim as optim
from torchvision import transforms

from torch.profiler import profile, record_function, ProfilerActivity

from data_processing.generator.crf_main_generator import create_generators
from utils.config import get_config

from model.architecture.retinanet import Retinanet
from model.architecture.vgg import Vggmax
from data_processing.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from model import nus_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--radar', help='Use radar modality?', type=bool, default=False)
    parser.add_argument('--load_path', help='Load model path', type=str, default=None)

    parser = parser.parse_args(args)

    # Create the data loaders
    ###### New DataLoaders here
    backbone = Vggmax(parser.radar)

    cfg = get_config('./config/default.cfg')
    train_generator, validation_generator, test_generator, test_night_generator, test_rain_generator = create_generators(cfg, backbone)
    dataloader = DataLoader(train_generator, batch_size=cfg.batchsize, num_workers=10)
    batch_size = cfg.batchsize
    print('=='*45)
    print('Length of train set: {}, val set: {}, test set: {}, test_night set: {}, test_rain set: {}'.format(len(train_generator), len(validation_generator), len(test_generator), len(test_night_generator), len(test_rain_generator)))
    print('=='*45)
    #####
    #print(train_generator[0][0][:,:,:,0])
    #exit(0)

    # Create the model
    image_size = (360, 640)
    if parser.radar:
        f_size = 254
    else:
        f_size = 256
    retinanet = Retinanet(backbone, pretrained=True, num_anchors=9, num_classes=train_generator.num_classes(), feature_size=f_size, image_size=image_size)

    if parser.load_path is not None:
        retinanet.load_state_dict(torch.load(parser.load_path, map_location='cuda:0'))
        start_ep = 20
    else:
        start_ep = 0
    
    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    #retinanet.train()
    #retinanet.module.freeze_bn()

    print('===='*7 + '\nNum training images: {}\n'.format(len(train_generator)) + '===='*10)
    start_time = time.time()
    
    for epoch_num in range(start_ep, parser.epochs):

        retinanet.train()
        #retinanet.module.freeze_bn()

        epoch_loss = []

        #for iter_num, data in enumerate(train_generator):
        for iter_num, data in enumerate(dataloader):#range(len(train_generator)):
            #try:
            #with profile(activities=[
            #            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            #    with record_function("model_inference"):
                    
            optimizer.zero_grad()
            #print('Data shape: {}, Annotation shape: {}'.format(data[0].shape, data[1][0].shape))
            #Data format data[0] = 5 Channel last image, data[1][0] = Regression annot, data[1][1] = classification annot.
            #data = train_generator[iter_num]
            #print(data[0].shape)
            #print(len(data[1]))
            #print(data[1][0].shape)
            #print(data[1][1].shape)

            if not parser.radar: # Crop and keep only image channels 
                img = data[0][0,:,:,:,:3]
            else:
                img = data[0][0,:,:,:,:]
            img = torch.permute(img.cuda().float(), (0,3,1,2))
            targets = get_annotations_for_batch(train_generator, iter_num, batch_size)
            #print('==='*10)
            #print(img[0,0,:,:])
            #print(torch.max(img[0,0,:,:]))
            #print('==='*10)
            #print(img[0,3,:,:])
            #print(torch.max(img[0,3,:,:]))
            #print('==='*10)
            #print(targets[0][:,:])
            #print(torch.max(targets[0][:,:]))
            #print('==='*10)
            #exit(0)
            '''
            ann = train_generator.load_annotations(iter_num)
            if (len(ann['labels']) == 0):
                targets = torch.tensor([[[-1, -1, -1, -1, -1]]]).cuda()
            else:
                #print(ann['bboxes'].shape, ann['labels'].shape)
                targets = np.hstack((ann['bboxes'], np.expand_dims(ann['labels'], axis=1))) 
                targets = torch.tensor(np.expand_dims(targets, axis=0)).cuda()
            '''

            classification_loss, regression_loss = retinanet([img, targets])
                
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss

            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            optimizer.step()

            loss_hist.append(float(loss))

            epoch_loss.append(float(loss))

            print(
                    'Ep: {} | Iter: {} | Cls loss: {:1.5f} | Reg loss: {:1.5f} | Running loss: {:1.5f} | Elp Time: {}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist), int(time.time()-start_time)), end='\r')

            #except Exception as e:
            #    print(e)
            #    continue
            del classification_loss, regression_loss
            #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        torch.save(retinanet.module.state_dict(), 'exp4_radar_image_retinanet_{}.pt'.format(epoch_num))
        #try:
        #    mAP = nus_eval.evaluate(train_generator, retinanet)
        #except Exception as e:
        #    print(e)
        #    continue

        scheduler.step(np.mean(epoch_loss))


    retinanet.eval()

    torch.save(retinanet, 'exp2_radar_image_model_final.pt')

def get_annotations_for_batch(generator, iter_num, batch_size):
    bstart = iter_num*batch_size
    btargets = []
    for i in range(batch_size):
        ann = generator.load_annotations(bstart+i)
        if (len(ann['labels']) == 0):
            btargets.append(torch.tensor([[-1., -1., -1., -1., -1.]]).cuda())
        else:
            #print(ann['bboxes'].shape, ann['labels'].shape)
            targets = np.hstack((ann['bboxes'], np.expand_dims(ann['labels'], axis=1))) 
            btargets.append(torch.tensor(targets).cuda())
            #targets = torch.tensor(np.expand_dims(targets, axis=0)).cuda()
    return btargets

if __name__ == '__main__':
    main()
