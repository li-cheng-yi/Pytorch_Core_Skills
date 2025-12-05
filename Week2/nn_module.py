import torch
from torch import nn
class NEUQ(nn.Module):
    def __init__(self):
        super(NEUQ,self).__init__()
    def forward(self,input):
        output=input+1
        return output
neuq=NEUQ()
x=torch.tensor(1.0)
output=neuq(x)
print(output)