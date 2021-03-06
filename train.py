import os
import json
import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer import Trainer
from utils import Logger


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, resume):
    train_logger = Logger()

    # setup data_loader instances
    data_loader = get_instance(module_data, 'data_loader', config)
    valid_data_loader = data_loader.split_validation()

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    print(model.gating_params)

    gating_params = next(x for i, x in enumerate(model.children()) if i == 4)
    gate_no = -1
    freeze = [(0,1,1), (0,2,1), (0,3,1), (0,4,1), (1,2,0), (1,3,1), (1,4,1), (2,3,1), (2,4,1), (3,4,1)] #after we get the latent's structure, otherwise ignore it
    
            
#     for gate in gating_params:
#         gate_no += 1
#         tensor_no = gate_no
#         for tensor in gate:
#             tensor_no += 1
#             if (gate_no, tensor_no) in freeze:
#                 value = freeze_values[freeze.index((gate_no, tensor_no))]
#                 print("Setting {}-{} to {}".format(gate_no, tensor_no, value))
#                 tensor.data = torch.Tensor(value)
#                 # tensor.requires_grad = False
#                 tensor = tensor.detach()

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer',
                             config, trainable_params)
    lr_scheduler = get_instance(
        torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

    trainer = Trainer(model, loss, metrics, optimizer,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger)
    fixed_structure = False #fixed_structure = True for fixed structure of the latent, otherwise set fixed_structure = False
    if fixed_structure:
      for (x, y, v) in freeze:
          print(model.gating_params[x][y - x -1])
          #trainer.model.gating_params[x][y - x -1].data = torch.Tensor([[v]]).detach()
          model.gating_params[x][y - x -1] = torch.autograd.variable([[v]]).detach()
          print("Setting {}-{} to {}".format(x, y, v))
          print(trainer.model.gating_params[x][y - x - 1].data)
        
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        # load config file
        with open(args.config) as handle:
            config = json.load(handle)
        # setting path to save trained models and log files
        path = os.path.join(config['trainer']['save_dir'], config['name'])

    elif args.resume:
        # load config from checkpoint if new config file is not given.
        # Use '--config' and '--resume' together to fine-tune trained model with changed configurations.
        config = torch.load(args.resume)['config']

        # with open(args.config) as handle:
        #     config = json.load(handle)

    else:
        raise AssertionError(
            "Configuration file need to be specified. Add '-c config.json', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
