import torch
import torchvision

model = torchvision.models.alexnet(pretrained=True, num_classes=10)
torchvision.models.resnet18(pretrained=True)

