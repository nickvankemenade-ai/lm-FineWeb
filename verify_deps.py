import torch
import numpy
import torch.nn
import torch.nn.functional
import torch.utils.data
import tqdm
import tokenizer
import time
import pathlib
import math


print(torch.__version__)
print(numpy.__version__)
print(tqdm.__version__)
print(tokenizer.__version__)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
