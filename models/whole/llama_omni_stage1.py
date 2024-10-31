import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        



    def forward(self, x):
        # input: audio tensor, text token

        # whisper encoder

        # adaptor

        # llama

        


    
if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    md = Model(None)
    # print(summary(md, torch.randn((8,64600)), show_input=False))
    op,res = md( torch.randn((8,64600)))
    print(op.shape)