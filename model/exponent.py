import numpy as np
import torch
import os
import pathlib
exponent = np.exp(-10000)
path = pathlib.Path.cwd()
print(os.path.join( path, "saved", "GraphVAE", "0124_153059","checkpoint-epoch200.pth"))