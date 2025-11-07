from datasets import load_dataset
from torchvision.ops import RoIPool
from transformers import DetrImageProcessor, DetrForObjectDetection, PreTrainedModel, Trainer, TrainingArguments, \
    BatchFeature, AutoImageProcessor, DetrConfig
from typing import List, Mapping, Union, Any, Tuple
import albumentations as A
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from torch import nn


# %%
class PreTrainingBackbone(PreTrainedModel):
    def __init__(self, config, backbone):
        super().__init__(config)
        self.backbone = backbone.model.backbone

        self.ffn = nn.Linear(256, 81)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.pooler = RoIPool(output_size=(7, 7), spatial_scale=1 / 32)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels, pixel_mask=None):
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=device)

        features_map = self.backbone(pixel_values, pixel_mask)[0][0][0]

        box_features = self.pooler(features_map, [label['boxes'] for label in labels])
        box_features = self.avgpool(box_features).squeeze(-1).squeeze(-1)
        logits = self.ffn(box_features)
        labels = torch.cat([label['class_labels'] for label in labels])
        loss = self.loss(logits, labels)

        return {"loss": loss}


def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data


def bbox_intersect(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    return x2 > x3 and x4 > x1 and y2 > y3 and y4 > y1


def divide_bboxes(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    divided_boxes = []
    if x3 > x1:
        divided_boxes.append([x1, y1, x3, y2])  # Left part
    if x4 < x2:
        divided_boxes.append([x4, y1, x2, y2])  # Right part
    if y3 > y1:
        divided_boxes.append([x1, y1, x2, y3])  # Top part
    if y4 < y2:
        divided_boxes.append([x1, y4, x2, y2])  # Bottom part
    return divided_boxes


def get_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def get_background_bboxes(background_bbox, handled_bbox, bboxes):
    if len(bboxes) != 0:
        bbox = bboxes[0]
        divided_bboxes = divide_bboxes(handled_bbox, bbox)
        for divided_bbox in divided_bboxes:
            if get_area(divided_bbox) > get_area(background_bbox):
                get_background_bboxes(background_bbox, divided_bbox,
                                      [bb for bb in bboxes if bbox_intersect(divided_bbox, bb)])
    else:
        if get_area(handled_bbox) > get_area(background_bbox):
            background_bbox[:] = handled_bbox


def format_image_annotations_as_coco(image_id, categories, areas, bboxes, sizes, bg_train=False):
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category + 1,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)
    if bg_train:
        background_bboxe = [0, 0, 0, 0]
        get_background_bboxes(background_bboxe, [0, 0, sizes[0], sizes[1]], bboxes)
        annotations.append({
            "image_id": image_id,
            "category_id": 0,  # Assuming 0 is the category ID for background
            "iscrowd": 0,
            "area": get_area(background_bboxe),
            "bbox": background_bboxe,
        })
    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def augment_and_transform_batch(examples, transform, image_processor, return_pixel_mask=False, bg_train=False):
    images = []
    annotations = []
    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))

        # apply augmentations
        output = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"], sizes=image.shape, bg_train=bg_train
        )
        annotations.append(formatted_annotations)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")

    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result


def main():
    config = DetrConfig()
    processor = DetrImageProcessor()
    detr_model = DetrForObjectDetection(config).cuda()

    # processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    # detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").cuda()
    # model = lambda *args, **kwargs: detr_model.model.backbone(*args, **kwargs)[0][0][0]

    pretrained_backbone = PreTrainingBackbone(detr_model.config, detr_model)

    dataset = load_dataset("detection-datasets/coco")

    train_augment_and_transform = A.Compose(
        [
            A.HorizontalFlip(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
    )
    validation_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
    )

    train_transform_batch = partial(
        augment_and_transform_batch, transform=train_augment_and_transform, image_processor=processor, bg_train=True
    )
    validation_transform_batch = partial(
        augment_and_transform_batch, transform=validation_transform, image_processor=processor, bg_train=True
    )

    dataset['train'] = dataset['train'].with_transform(train_transform_batch)
    dataset['val'] = dataset['val'].with_transform(validation_transform_batch)

    training_args = TrainingArguments(remove_unused_columns=False, report_to='wandb', auto_find_batch_size=True,
                                      do_eval=True)

    trainer = Trainer(
        args=training_args,
        model=pretrained_backbone,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        processing_class=processor,
        data_collator=collate_fn
    )
    trainer.train()
    trainer.evaluate()

    train_transform_batch = partial(
        augment_and_transform_batch, transform=train_augment_and_transform, image_processor=processor, bg_train=True
    )
    validation_transform_batch = partial(
        augment_and_transform_batch, transform=validation_transform, image_processor=processor, bg_train=True
    )

    dataset = load_dataset("detection-datasets/coco")

    dataset['train'] = dataset['train'].with_transform(train_transform_batch)
    dataset['val'] = dataset['val'].with_transform(validation_transform_batch)

    training_args = TrainingArguments(remove_unused_columns=False, report_to='wandb', num_train_epochs=50,
                                      auto_find_batch_size=True, do_eval=True)

    detr_model.model.backbone = pretrained_backbone.backbone
    trainer = Trainer(
        args=training_args,
        model=detr_model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        processing_class=processor,
        data_collator=collate_fn
    )
    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()
