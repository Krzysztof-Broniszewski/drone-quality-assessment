import torch
import torch.nn as nn
import torchvision.models as models

class MultiHeadCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(MultiHeadCNN, self).__init__()
        base = models.resnet18(pretrained=True)
        in_features = base.fc.in_features
        base.fc = nn.Identity()

        self.backbone = base
        self.ostrosc = nn.Linear(in_features, num_classes)
        self.swiatlo = nn.Linear(in_features, num_classes)
        self.ekspozycja = nn.Linear(in_features, num_classes)
        self.kadr = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return {
            "ostrosc": self.ostrosc(x),
            "swiatlo": self.swiatlo(x),
            "ekspozycja": self.ekspozycja(x),
            "kadr": self.kadr(x)
        }
