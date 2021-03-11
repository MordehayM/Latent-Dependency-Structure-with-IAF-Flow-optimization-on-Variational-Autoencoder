import torch

def map_compare(output, target):
    mu = output['mu_0']
    map_output = (mu > 0.5).type(torch.float)
    return torch.mean((map_output == target).type(torch.float))