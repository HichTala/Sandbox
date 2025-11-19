import torch
from torch import nn
from torchvision.ops import RoIPool
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput


class PreTrainingBackboneForImageClassification(PreTrainedModel):
    def __init__(self, config, model):
        super().__init__(config)
        self.backbone = model.model.backbone

        self.ffn = nn.Linear(256, 81)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.pooler = RoIPool(output_size=(7, 7), spatial_scale=1 / 32)

    def forward(self, pixel_values, labels, pixel_mask=None, **kwargs):
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=device)

        features_map = self.backbone(pixel_values, pixel_mask)[0][0][0]

        box_features = self.pooler(features_map, [label['boxes'] for label in labels])
        box_features = self.avgpool(box_features).squeeze(-1).squeeze(-1)
        logits = self.ffn(box_features)
        labels = torch.cat([label['class_labels'] for label in labels])
        loss = self.loss_function(labels, logits, self.config, **kwargs)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits
        )