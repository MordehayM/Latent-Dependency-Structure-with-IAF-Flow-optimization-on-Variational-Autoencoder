import os
import json
import argparse
import torch
from torch import nn
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer import Trainer
from utils import Logger
import torchvision



torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, path):
    train_logger = Logger()

    # setup data_loader instances
    data_loader = get_instance(module_data, 'data_loader', config)
    valid_data_loader = data_loader.split_validation()
    model = get_instance(module_arch, 'arch', config)
    # build model architecture
    n_train = False
    model = get_instance(module_arch, 'arch', config)
    
    device = torch.device("cpu")
    checkpoint = torch.load(path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    num_images = 64
    images = model.sample(num_images=num_images, device=device)
    #range = images.max(dim=1, keepdim=True).values - images.min(dim=1, keepdim=True).values
    #normalize_images = (images - images.min(dim=1, keepdim=True).values) / range
    #binary_images = (normalize_images > 0.5).type(torch.float)
    images = images.reshape(-1, 1, 28, 28)
    #torch.reshape(images,(-1, 1, 28, 28))
    torchvision.utils.save_image(torchvision.utils.make_grid(images), './samples/sample_image.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-r', '--path', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')           
    args = parser.parse_args()



    if args.path:
        # load config from checkpoint if new config file is not given.
        # Use '--config' and '--path' together to fine-tune trained model with changed configurations.
        config = torch.load(args.path)['config']

        # with open(args.config) as handle:
        #     config = json.load(handle)

    else:
        raise AssertionError(
            "Path need to be specified")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.path)
