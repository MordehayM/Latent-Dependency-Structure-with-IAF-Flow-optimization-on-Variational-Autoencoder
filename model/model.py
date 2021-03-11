import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.distributions as D
from base import BaseModel
import flow

EPSILON = 1e-30


def log_gaussian(x, mu, var):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and var evaluated at x.
    :param x: point to evaluate
    :param mu: mean of distribution
    :param var: variance of distribution
    :return: log N(x|mu,var)
    """
    log_pdf = - 0.5 * torch.log(2 * torch.tensor([math.pi], device=x.device)) - torch.log(var + 1e-8) / 2 - (x - mu) ** 2 / (2 * var + 1e-8)
    # print('Size log_pdf:', log_pdf.shape)
    return log_pdf

class GraphVAE(BaseModel):
    def __init__(self, input_dim, n_nodes, node_dim, flow_depth):
        super(GraphVAE, self).__init__()
        # store parameters
        self.input_dim = input_dim
        self.n_nodes = n_nodes
        self.node_dim = node_dim

        # encoder: x -> h_x
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, 128)
        )
        # bottom-up inference: predicts parameters of P(z_i | x)
        self.bottom_up = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.Linear(128, node_dim),
                nn.Linear(node_dim, 3*node_dim) # split into mu and logvar and h
            )
        for _ in range(n_nodes)]) # ignore z_n



        # top-down inference: predicts parameters of P(z_i | Pa(z_i))
        self.top_down = nn.ModuleList([
            nn.Sequential(
                nn.Linear((n_nodes - i - 1)*node_dim, 128), # parents of z_i are z_{i+1} ... z_N
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.Linear(128, node_dim),
                nn.Linear(node_dim, 2*node_dim) # split into mu and logvar
            )
        for i in range(n_nodes-1)]) # ignore z_n

        # decoder: (z_1, z_2 ... z_n) -> parameters of P(x)
        self.decoder = nn.Sequential(
            nn.Linear(node_dim*n_nodes, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, input_dim)
        )

        # mean of Bernoulli variables c_{i,j} representing edges
        self.gating_params = nn.ParameterList([
            nn.Parameter(torch.empty(n_nodes - i - 1, 1, 1).fill_(0.5), requires_grad=True) #for fixed structure requires_grad=False, other-wise requires_grad=True
        for i in range(n_nodes-1)]) # ignore z_n


        # distributions for sampling
        self.unit_normal = D.Normal(torch.zeros(self.node_dim), torch.ones(self.node_dim))
        self.gumbel = D.Gumbel(0., 1.)

        # other parameters / distributions
        self.tau = 1.0 #decreased to 0.99^epoch at each epoch

        hidden_size = node_dim * 2 #hidden size of the InverseAutoregressiveFlow

        modules = []
        for _ in range(flow_depth):
            modules.append(flow.InverseAutoregressiveFlow(num_input=node_dim,
                                                          num_hidden=hidden_size,
                                                          num_context=node_dim))
            modules.append(flow.Reverse(node_dim))
        self.IAFs = nn.ModuleList([flow.FlowSequential(*modules) for _ in range(n_nodes)]) # FlowSequential for each node z_n(yielding in z_nT)

        #self.IAFs_m = nn.ModuleList([nn.Sequential(*modules) for _ in range(n_nodes)])
        #self.IAFs = nn.ModuleList([flow.FlowSequential(*self.IAFs_m[i]) for i in range(n_nodes)])
    def forward(self, x, n_samples=1): #n_sample for the likelihood calculation at the test phase
        # x: (batch_size, input_size)
        output = {}
        for sample in range(n_samples):
            
            x = torch.reshape(x, (-1,784))
            hx = self.encoder(x)

            # sample z_n from N(0, I)
            #z_pp= self.unit_normal.sample([x.size(0)]).to(x.device)
            #parents = [z_n]
            mu_z = []
            sigma_z = []
            bu = self.bottom_up[-1](hx)
            mu_bu, sigma_bu, h_bu = bu[:, :self.node_dim], F.softplus(bu[:, self.node_dim:self.node_dim*2]),\
                              bu[:, self.node_dim*2:]
            z0 = mu_bu + sigma_bu * self.unit_normal.sample([x.size(0)]).to(x.device)
            log_z_x = [log_gaussian(z0, mu_bu, sigma_bu**2)]
            z0_T, total_log_prob = self.IAFs[-1](input=z0.unsqueeze(1), context=h_bu.unsqueeze(1))
            z0_T = z0_T.squeeze(1)
            total_log_prob = total_log_prob.squeeze(1)
            log_z = [log_gaussian(z0_T, torch.tensor([0], device=x.device), torch.tensor([1], device=x.device))]
            det_tot = [total_log_prob]
            parents = [z0_T]
            #parents = [z_pp]
            #mu_z.append(mu_bu) 
            #sigma_z.append(sigma_bu)

            #mu_z_prior = [torch.zeros(x.size(0), self.node_dim).to(x.device)]  # [B x node_dim]
            #sigma_z_prior = [torch.ones(x.size(0), self.node_dim).to(x.device)]  # [B x node_dim]

            #mu_bu_l = [torch.zeros(x.size(0), self.node_dim).to(x.device)]  # [B x node_dim]
            #sigma_bu_l = [torch.ones(x.size(0), self.node_dim).to(x.device)]  # [B x node_dim]

            for i in reversed(range(self.n_nodes-1)):
                
                #c = self.gating_params[i].data  #for fixed structure, other-wise this line is commented
                
                #print(self.gating_params[i].data)
                
                #for fixed structure the following section(marked with triple '#') is omited:
                ### - start of omission
                #'''
                self.gating_params[i].data = self.gating_params[i].data.clamp(0., 1.)
                # compute gating constants c_{i,j}
                mu = self.gating_params[i]
                eps1, eps2 = self.gumbel.sample(mu.size()).to(x.device), self.gumbel.sample(mu.size()).to(x.device)
                num = torch.exp((eps2 - eps1)/self.tau)
                t1 = torch.pow(mu, 1./self.tau)
                t2 = torch.pow((1.-mu), 1./self.tau)*num
                c = t1 / (t1 + t2 + EPSILON)
                if torch.isnan(t1).any() or torch.isnan(t2).any() or torch.isnan(c).any() or torch.isnan(mu).any():
                    print(t1,t2,c,mu)
                #'''
                ### - end of omission
                
                # find concatenated parent vector
                parent_vector = (c * torch.stack(parents)).permute(1,0,2).reshape(x.size(0), -1) #same c for the node_dim, however it's changed per node
                # top-down inference
                td = self.top_down[i](parent_vector)
                mu_td, sigma_td = td[:, :self.node_dim], F.softplus(td[:, self.node_dim:])

                #mu_z_prior.append(mu_td)
                #sigma_z_prior.append(sigma_td)

                # bottom-up inference
                bu = self.bottom_up[i](hx)
                mu_bu, sigma_bu, h_bu = bu[:, :self.node_dim], F.softplus(bu[:, self.node_dim:self.node_dim * 2]), \
                                        bu[:, self.node_dim * 2:]

                #mu_bu_l.append(mu_bu)
                #sigma_bu_l.append(sigma_bu)

                # precision weighted fusion
                mu_zi = (mu_td * sigma_bu**2 + mu_bu * sigma_td**2) / (sigma_td**2 + sigma_bu**2 + EPSILON)
                sigma_zi = (sigma_bu * sigma_td) / (torch.sqrt(sigma_td**2 + sigma_bu**2) + EPSILON)#B x 1
                # sample z_i from P(z_i | pa(z_i), x)
                z_i = mu_zi + sigma_zi * self.unit_normal.sample([x.size(0)]).to(x.device)
                log_z_x.append(log_gaussian(z_i, mu_zi, sigma_zi ** 2))
                zi_T, total_log_prob = self.IAFs[i](input=z_i.unsqueeze(1), context=h_bu.unsqueeze(1))
                zi_T = zi_T.squeeze(1)
                total_log_prob = total_log_prob.squeeze(1)
                #det_tot += total_log_prob

                det_tot.append(total_log_prob)
                #log_z += log_gaussian(zi_T, mu_td, sigma_td**2)
                log_z.append(log_gaussian(zi_T, mu_td, sigma_td**2)) # store samples and parameters
                parents.append(zi_T)
                #mu_z.append(mu_zi)
                #sigma_z.append(sigma_zi)


            # sample from approximate posterior distribution q(z_1, z_2 ... z_n|x)
            #z = torch.ones(x.size(0), self.node_dim).to(x.device)
            #for i in parents:
            #    z = z*i
            z = torch.cat(parents, dim=1) #concatenation over the nodes
            #z = parents[-1]
            out = torch.sigmoid(self.decoder(z))

            # build output
            
            output[f'mu_{sample}'] = out
            output[f'det_{sample}'] = det_tot
            output[f'log_z_{sample}'] = log_z
            output[f'log_z_x_{sample}'] = log_z_x

        '''
        output['means'] = mu_z
        output['sigmas'] = sigma_z
        output['means_prior'] = mu_z_prior
        output['sigmas_prior'] = sigma_z_prior
        output['means_bu'] = mu_bu_l
        output['sigmas_bu'] = sigma_bu_l
        '''
        
        # output['gate_params'] = self.gating_params.detach()
        return output
        
    def sample(self, num_images, device):
        z_n = self.unit_normal.sample([num_images]).to(device)
        parents = [z_n]
        for i in reversed(range(self.n_nodes-1)):
            # compute gating constants c_{i,j}
            c = self.gating_params[i].data
            
            '''
            mu = self.gating_params[i] #[num of parents x 1 x 1]
            eps1, eps2 = self.gumbel.sample(mu.size()).to(device), self.gumbel.sample(mu.size()).to(device)
            num = torch.exp((eps2 - eps1)/self.tau)
            #num = torch.exp((eps2 - eps1) / self.tau)
            t1 = torch.pow(mu, 1./self.tau)
            t2 = torch.pow((1.-mu), 1./self.tau)*num
            c = t1 / (t1 + t2 + EPSILON) #[num of parents x 1 x 1]
            if torch.isnan(t1).any() or torch.isnan(t2).any() or torch.isnan(c).any() or torch.isnan(mu).any():
                print(t1,t2,c,mu)
            '''
            # find concatenated parent vector
            parent_vector = (c * torch.stack(parents)).permute(1, 0, 2).reshape(num_images, -1)  # [num_images x num of parents], this is the sampling action
            # top-down inference
            td = self.top_down[i](parent_vector)
            mu_td, sigma_td = td[:, :self.node_dim], F.softplus(td[:, self.node_dim:])
            z_i = mu_td + sigma_td * self.unit_normal.sample([num_images]).to(device)
            parents.append(z_i)

        #z = torch.ones(num_images, self.node_dim).to(device)
        #for i in parents:
        #    z = z * i
        #z = parents[-1]
        z = torch.cat(parents, dim=1)
        out = torch.sigmoid(self.decoder(z))
        return out

