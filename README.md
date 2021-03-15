# Latent-Dependency-Structure-with-IAF-Flow-optimization-on-Variational-Autoencoder
### By Reut Benaim and Mordehay Moradi

In this project, we propose to improve VAE performance.
The project has been compounded by both adding hierarchical latent dependencies
and building an inference network with normalizing flow.
We use the methods from the paper ["Variational Autoencoders with Jointly Optimized
Latent Dependency Structure (Jiawei He1∗ & Yu Gong1, et al)"](https://openreview.net/forum?id=SJgsCjCqt7) that suggests learning these latent dependencies,
rather than using predefined models with potentially limited performance,
and the paper ["Improved Variational Inference with Inverse Autoregressive Flow" (Diederik P. Kingma, et al)](https://arxiv.org/abs/1606.04934
) 
that suggests a new type of normalizing flow framework, inverse autoregressive flow (IAF),
which improves on the diagonal Gaussian approximate posteriors and scales well to high-dimensional latent space.

## Initial setup

Clone the repository
```bash
git clone https://github.com/MordehayM/Latent-Dependency-Structure-with-IAF-Flow-optimization-on-Variational-Autoencoder.git
```
Install the packages that appear in the requirements.txt file 

## Usage

**Order of operations:**

| Description | Command |
| --- | --- |
| Mnsit download | `python mnist_create.py` |
| Train - latent dependencies is learned - epochs~200 | `python train.py --config config.json` |
| Train - latent dependencies is fixed - epochs~800, set freeze variable in train.py file and change require_grad=False to the gating variable in model.py file. | `python train.py --config config.json` |
| Resume from checkpoint | `python train.py --resume <path_to_checkpoint>` |
| Using multiple GPUs (equivalent to `"CUDA_VISIBLE_DEVICES=2 python train.py -c config.py"`) | `python train.py --device 2 -c config.json` |
| Test | `python test.py --path <path_to_checkpoint>` |
| Sample - create new samples | `python test.py --path <path_to_checkpoint>` |
| Visualization | `tensorboard --logdir <path_to_log_dir>` |
  
- Train - latent dependencies is learned - epochs~200
  
  `python train.py --config config.json`
  
- Train - latent dependencies is fixed- epochs~800, set freeze variable in train.py file
  and change require_grad=False to the gating variable in model.py file.
  
  `python train.py --config config.json`
- Resume from checkpoint
  
  `python train.py --resume <path_to_checkpoint>`
- Using multiple GPUs (equivalent to `"CUDA_VISIBLE_DEVICES=2 python train.py -c config.py"`)
  
  `python train.py --device 2 -c config.json`
- Test
  
  `python test.py --path <path_to_checkpoint>`
- Sample - create new samples
  
  `python sample.py --path <path_to_checkpoint>`
- Visualization 
  
  `tensorboard --logdir <path_to_log_dir>`

## Configuration

The config file is specified in JSON format. Modify the file in accordance to your analysis(nodes number, dim size etc) 

## Results and explanation

A report on Latent Dependency Structure with IAF Flow optimization on Variational Autoencoder is given.

## References

The code was written inspired by the following code URLs https://github.com/ys1998/vae-latent-structure
&& https://github.com/altosaar/variational-autoencoder







