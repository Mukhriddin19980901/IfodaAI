from torch import nn
from torchvision import models
import torch
from json import JSONEncoder

class ResNetModel(nn.Module):
    def __init__(self,num_classes = 5 , pretrained = True):
        super(ResNetModel,self).__init__()
        self.resnet = models.resnet18(pretrained = pretrained) # resnet50 modelini weight va biaslarini saqlagan xolda pretraining uchun olamiz

        # Faqat fine-tune qilganim uchun barcha parametrlarni muzlatib qoydim
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Oxirgi classifierni o'rnini o'zimizni quyidagi arxitektura bilan almashtirish
        in_features = self.resnet.fc.in_features 

        # Fine-tuning qilamiz
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features , 512), #feautes bizda kiruvchi data parametrlari
            nn.BatchNorm1d(512),  
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),  # datani stabillashtiradi
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256,num_classes) 
        )
    def forward(self,x):
        return self.resnet(x)
def accuracy(outputs, labels):
    # outputs: [B, num_classes], labels: [B]
    preds = outputs.argmax(dim=1)
    correct = (preds == labels).float().sum()
    return correct / labels.size(0)
    
class TensorEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            # Move to CPU, detach from graph, convert to Python list
            return obj.cpu().detach().numpy().tolist()
        # Let the base class raise the TypeError for other types
        return super().default(obj)