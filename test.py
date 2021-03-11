import os
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import get_instance
import numpy as np

EPSILON = 1e-30

def log_likelihood_bernoulli(x, mu):
	one_mask = (x == 1.).type(torch.float)
	zero_mask = (x == 0.).type(torch.float)
	return torch.sum(torch.log(mu + EPSILON)*one_mask + torch.log(1.-mu + EPSILON)*zero_mask, dim=1)





def main(config, resume):
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    # model.summary()

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns), device=device)
    num_sample = 100
    log_p_x = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            # target = torch.reshape(target[0],(-1, 784))
            data = data.to(device)  # , target.to(device)
            output = model(data, num_sample)
            #
            # save sample images, or do something with output here
            #

            # for i in range(10):
            #  if torch.equal(target[i], target[i+1]):
            #    print('true')

            # computing loss, metrics on test set
            print(output['mu_0'].shape)
            print(data.shape)

            loss = loss_fn(output, data)  # returned as tuple
            loss = loss[0] + loss[1]
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, data) * batch_size
            log_p_x_l = []
            for i in range(num_sample):
                mu = output[f'mu_{i}']
                l1 = log_likelihood_bernoulli(data, mu)
                l2 = torch.sum(-sum(output[f'det_{i}']) + sum(output[f'log_z_{i}']) - sum(output[f'log_z_x_{i}']), dim=1)
                log_p_x_l.append(l1+l2)
            log_p_x_l = torch.stack(log_p_x_l, dim=1)
            log_p_x += (torch.logsumexp(log_p_x_l, dim=1) - np.log(num_sample)).sum()




    n_samples = len(data_loader.sampler) #check!! = 10,000
    print(n_samples)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    log.update({'log_p_x': log_p_x / n_samples})
    print(log)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)