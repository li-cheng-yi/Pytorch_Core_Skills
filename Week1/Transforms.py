from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
#python的用法-》tensor数据类型

img_path="C:\\Users\\60700\\PycharmProjects\\Pytorch_Core_Skills\\dataset\\train\\ants_image\\0013035.jpg"
img=Image.open(img_path)
writer=SummaryWriter("logs")
tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)
writer.add_image("Tensor_image",tensor_img)
writer.close()
