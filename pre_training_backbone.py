import torch
from matplotlib import pyplot as plt, patches
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

        images = pixel_values.permute(0, 2, 3, 1).cpu().numpy()
        orig_sizes = [torch.cat([label['size'], label['size']]).cpu().numpy() for label in labels]
        bboxes = [label['boxes'].cpu().numpy() * orig_size for label, orig_size in zip(labels, orig_sizes)]
        class_labels = [label['class_labels'].cpu().numpy() for label in labels]

        def plot(i):
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(images[i])

            for bbox, label in zip(bboxes[i], class_labels[i]):
                cx, cy, w, h = bbox
                x1 = cx - w / 2
                y1 = cy - h / 2
                # w, h = x2 - x1, y2 - y1
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                         edgecolor="red", facecolor="none")
                ax.add_patch(rect)
                ax.text(x1, y1, label, color="white", fontsize=10,
                        bbox=dict(facecolor="red", alpha=0.5))
            plt.show()

        features_map = self.backbone(pixel_values, pixel_mask)[0][0][0]

        h_feat, w_feat = features_map.shape[-2:]
        rois = [
            torch.cat([
                label['boxes'] * torch.tensor([w_feat, h_feat, w_feat, h_feat], device=features_map.device),
            ], dim=1)
            for label in labels
        ]
        box_features = self.pooler(features_map, rois)
        box_features = self.avgpool(box_features).squeeze(-1).squeeze(-1)
        logits = self.ffn(box_features)
        labels = torch.cat([label['class_labels'] for label in labels])
        loss = self.loss_function(labels, logits, self.config, **kwargs)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits
        )
