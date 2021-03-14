import torch

EPSILON = 1e-30

def log_likelihood_bernoulli(x, mu):
	one_mask = (x == 1.).type(torch.float)
	zero_mask = (x == 0.).type(torch.float)
	return torch.sum(torch.log(mu + EPSILON)*one_mask + torch.log(1.-mu + EPSILON)*zero_mask, dim=1).mean()

def kl_divergence_normal(mu, sigma, mu_prior, sigma_prior):
	return torch.sum((sigma**2 + (mu-mu_prior)**2)/(2*(sigma_prior**2)) + torch.log(sigma_prior/sigma + EPSILON) - 0.5, dim=1).mean()

def loss_MNIST(output, target):
	mu = output['mu_0']
	l1 = log_likelihood_bernoulli(target, mu) #log_p(x/z_T)
	l2 = torch.sum(-sum(output['det_0']) + sum(output['log_z_0']) - sum(output['log_z_x_0']), dim=1).mean()
	return -l2, -l1 # log_p(x/z_T)+log_p(z_T/c)-log_q(x/z_T)+sum(log_det(|dz_t/dz_t-1|), t=1..T)
 