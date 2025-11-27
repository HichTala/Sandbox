import numpy as np
import torch
from matplotlib import pyplot as plt, patches
from torch import nn
from torchvision.ops import RoIPool, MultiScaleRoIAlign, RoIAlign
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput


class PreTrainingBackboneForImageClassification(PreTrainedModel):
    def __init__(self, config, model):
        super().__init__(config)
        self.backbone = model.model.backbone.conv_encoder.model

        self.ffn = nn.Linear(2048, 81)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.pooler = RoIAlign((7,7), spatial_scale=1/32, sampling_ratio=2)
        # self.pooler = MultiScaleRoIAlign(
        #     featmap_names=['0', '1', '2', '3'],  # depending on backbone outputs
        #     output_size=7,
        #     sampling_ratio=2
        # )


    def forward(self, pixel_values, labels, pixel_mask=None, **kwargs):
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

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

        # plot(0)
        # plot(1)
        # plot(2)
        # plot(3)

        # features_map = self.backbone(pixel_values, pixel_mask)[0][0][0]
        features_maps = self.backbone(pixel_values)
        features_maps = {str(i): features_maps[i] for i in range(len(features_maps))}

        h_feat, w_feat = features_maps['3'].shape[-2:]
        rois = [
            torch.cat([
                label['boxes'] * torch.tensor([w_feat, h_feat, w_feat, h_feat], device=device),
            ], dim=1)
            for label in labels
        ]
        box_features = self.pooler(features_maps['3'], rois)
        # with torch.no_grad():
        #     feats = self.pooler(features_maps['3'], rois)
        #     print("pooled mean:", feats.mean().item())
        #     print("pooled std:", feats.std().item())
        box_features = self.avgpool(box_features).squeeze(-1).squeeze(-1)
        logits = self.ffn(box_features)
        labels = torch.cat([label['class_labels'] for label in labels])

        # probs = logits.softmax(dim=1)
        # print("logits mean/std:", logits.mean().item(), logits.std().item())
        # print("probs mean/std:", probs.mean().item(), probs.std().item())
        # preds = probs.argmax(dim=1)
        # print("unique preds:", torch.unique(preds, return_counts=True))
        # print("unique labels:", torch.unique(labels, return_counts=True))

        loss = self.loss_function(labels, logits, self.config, **kwargs)

        def normalize_map(x):
            """Normalize a 2D map to [0,1]."""
            x = x - x.min()
            x = x / (x.max() + 1e-5)
            return x

        # plt.imshow(normalize_map(np.mean(features_maps['3'][0][0][0][0].cpu().numpy(), axis=0)), cmap="viridis")
        # plt.show()
        # plt.imshow(normalize_map(np.mean(features_maps['3'][0][1][0][0].detach().cpu().numpy(), axis=0)), cmap="viridis")
        # plt.show()
        # plt.imshow(normalize_map(np.mean(features_maps['3'][0][2][0][0].detach().cpu().numpy(), axis=0)), cmap="viridis")
        # plt.show()
        # plt.imshow(normalize_map(np.mean(features_maps['3'][0][3][0][0].detach().cpu().numpy(), axis=0)), cmap="viridis")
        # plt.show()

        # plt.imshow(normalize_map(np.mean(features_maps['3'][0].detach().cpu().numpy(), axis=0)), cmap="viridis")
        # plt.show()
        def plot(i, image):
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(normalize_map(np.mean(image[i].detach().cpu().numpy(), axis=0)))

            for bbox, label in zip(rois[i].cpu().numpy(), class_labels[i]):
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

        # plot(0, features_maps['3'])
        # plot(0, features_maps['2'])
        # plot(0, features_maps['1'])
        # plot(0, features_maps['0'])
        # plot(1, features_maps['3'])
        # plot(1)
        # plot(2)
        # plot(3)


        return ImageClassifierOutput(
            loss=loss,
            logits=logits
        )
