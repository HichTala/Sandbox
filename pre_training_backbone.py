from torch import nn
from torchvision.ops import RoIAlign
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput


class PreTrainingBackboneForImageClassification(PreTrainedModel):
    def __init__(self, config, model):
        super().__init__(config)
        self.backbone = model.model.backbone.conv_encoder.model
        # self.backbone = model.resnet

        self.ffn = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 81)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.pooler = RoIAlign((7, 7), spatial_scale=1 / 32, sampling_ratio=2)

    def forward(self, pixel_values, labels, pixel_mask=None, **kwargs):
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device
        features_maps = self.backbone(pixel_values)
        logits = self.ffn(features_maps[-1])
        loss = self.loss_function(labels, logits, self.config, **kwargs)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits
        )
