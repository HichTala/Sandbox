import torch
from torch import nn
from transformers.modeling_outputs import ImageClassifierOutput


class PreTrainingBackboneForImageClassification(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.model.model[:4]

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(64, 1000)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
