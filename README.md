# Latent-Dependency-Structure-with-IAF-Flow-optimization-on-Variational-Autoencoder

In this project, we propose to improve VAE performance.
The project has been compounded by both adding hierarchical latent dependencies
and building an inference network with normalizing flow.
We use the methods from the paper "VARIATIONAL AUTOENCODERS WITH JOINTLY OPTIMIZED
LATENT DEPENDENCY STRUCTURE" (Jiawei He1∗ & Yu Gong1, n.d.) that suggests learning these latent dependencies,
rather than using predefined models with potentially limited performance,
and the paper "Improved Variational Inference with Inverse Autoregressive Flow" (Diederik P. Kingma, n.d.) 
that suggests a new type of normalizing flow framework, inverse autoregressive flow (IAF),
which improves on the diagonal Gaussian approximate posteriors and scales well to high-dimensional latent space.

## Initial setup
Clone the repository
```bash
git clone https://github.com/MordehayM/Latent-Dependency-Structure-with-IAF-Flow-optimization-on-Variational-Autoencoder.git
```
Install the packages that appear in the requirements.txt file 

## Usage
```bash
Order of operations:

#mnsit download
  python mnist_create.py
# train - latent dependencies is learned - epochs~200
  python train.py --config config.json
# train - latent dependencies is fixed- epochs~1000, set freeze variable in train.py file
# and change require_grad=False to the gating variable
  python train.py --config config.json  
# resume from checkpoint
  python train.py --resume <path_to_checkpoint>
# using multiple GPUs (equivalent to "CUDA_VISIBLE_DEVICES=2 python train.py -c config.py")
  python train.py --device 2 -c config.json 
# test
  python test.py --resume <path_to_checkpoint>
# sample - crete new samples
  python sample.py --resume <path_to_checkpoint>
# visualization 
  tensorboard --logdir <path_to_log_dir>
```
## Configuration
The config file is specified in JSON format. Modify the file in accordance with your analysis(nodes number, dim size etc) 

## References
The code was written inspired by the following code URLs https://github.com/ys1998/vae-latent-structure
&& https://github.com/altosaar/variational-autoencoder













