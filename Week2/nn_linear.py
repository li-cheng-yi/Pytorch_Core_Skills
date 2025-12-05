import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../Week2/data",transform=torchvision.transforms.ToTensor(),train=False,download=True)
dataloader=DataLoader(dataset,64)
class NEUQ(nn.Module):
    def __init__(self):
        super(self,NEUQ).__init__()
        self.linear1=Linear(196608,10)
    def forward(self,input):
        output=self.linear1(input)
        return output
neuq=NEUQ()

writer=SummaryWriter("../Week2/logs_linear")
step=0
for data in dataloader:
    imgs,targets=data
    output=torch.flatten(imgs)
    writer.add_images("input",imgs,step)
    writer.add_images("output",output,step)
    step+=1


