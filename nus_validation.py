import argparse
import torch
from torchvision import transforms

from data_processing.dataloader import CocoDataset, Resizer, Normalizer
from data_processing.generator.crf_main_generator import create_generators
from utils.config import get_config
from model import nus_eval

from model.architecture.retinanet import Retinanet
from model.architecture.vgg import Vggmax

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple validation script for validating a RetinaNet network.')

    parser.add_argument('--model_path', help='Path to model', type=str)
    parser.add_argument('--radar', type=bool, default=False)

    parser = parser.parse_args(args)

    backbone = Vggmax(radar=parser.radar)
    cfg = get_config('./config/default.cfg')
    train_generator, validation_generator, test_generator, test_night_generator, test_rain_generator = create_generators(cfg, backbone)

    # Create the model
    image_size = (360, 640)
    if parser.radar:
        fsize = 254
    else:
        fsize = 256
    retinanet = Retinanet(backbone, num_anchors=9, num_classes=validation_generator.num_classes(), feature_size=fsize, image_size=image_size)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    #retinanet.module.freeze_bn()

    nus_eval.evaluate(validation_generator, retinanet)


if __name__ == '__main__':
    main()
