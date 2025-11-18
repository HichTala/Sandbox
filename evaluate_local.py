from dataclasses import dataclass

from datasets import load_dataset
from torchmetrics.detection._mean_ap import MeanAveragePrecision
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
from transformers.image_transforms import center_to_corners_format


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor





def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes


@torch.no_grad()
def compute_metrics(
        evaluation_results,
        image_processor: AutoImageProcessor,
        threshold: float = 0.0,
        id2label=None,
):
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        # collect image sizes, we will need them for predictions post processing
        batch_image_sizes = torch.tensor([x["orig_size"] for x in batch])
        image_sizes.append(batch_image_sizes)
        # collect targets in the required format for metric computation
        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()-1] if id2label is not None else class_id.item()
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

    return metrics


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
    config = DetrConfig(num_labels=80)
    processor = DetrImageProcessor()
    detr_model = DetrForObjectDetection(config).cuda()

    # processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    # detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").cuda()
    # model = lambda *args, **kwargs: detr_model.model.backbone(*args, **kwargs)[0][0][0]

    # pretrained_backbone = PreTrainingBackbone(detr_model.config, detr_model)

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

    # training_args = TrainingArguments(remove_unused_columns=False, report_to='wandb', auto_find_batch_size=True,
    #                                   do_eval=True, num_train_epochs=1)

    # trainer = Trainer(
    #     args=training_args,
    #     model=pretrained_backbone,
    #     train_dataset=dataset['train'],
    #     eval_dataset=dataset['val'],
    #     processing_class=processor,
    #     data_collator=collate_fn
    # )
    # trainer.train()
    # trainer.evaluate()

    train_transform_batch = partial(
        augment_and_transform_batch, transform=train_augment_and_transform, image_processor=processor, bg_train=False
    )
    validation_transform_batch = partial(
        augment_and_transform_batch, transform=validation_transform, image_processor=processor, bg_train=False
    )

    dataset = load_dataset("detection-datasets/coco")

    dataset['train'] = dataset['train'].with_transform(train_transform_batch)
    dataset['val'] = dataset['val'].with_transform(validation_transform_batch).select(range(500))

    categories = dataset["train"].features["objects"]["category"].feature.names
    id2label = dict(enumerate(categories))
    label2id = {v: k for k, v in id2label.items()}

    training_args = TrainingArguments(remove_unused_columns=False, report_to='wandb', num_train_epochs=50,
                                      per_device_eval_batch_size=2, do_eval=True, eval_do_concat_batches=False)
    eval_compute_metrics_fn = partial(
        compute_metrics, image_processor=processor, id2label=id2label, threshold=0.0
    )

    # detr_model.model.backbone = pretrained_backbone.backbone
    trainer = Trainer(
        args=training_args,
        model=detr_model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        processing_class=processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn
    )
    # trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()
