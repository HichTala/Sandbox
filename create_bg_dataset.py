import os

from datasets import load_dataset
from tqdm import tqdm


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
        divided_boxes.append([x1, y4, x2, y2])  #Bottom part
    return divided_boxes

def area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def get_background_bboxes(background_bbox, handled_bbox, bboxes):
    if len(bboxes) != 0:
        bbox = bboxes[0]
        divided_bboxes = divide_bboxes(handled_bbox, bbox)
        for divided_bbox in divided_bboxes:
            if area(divided_bbox) > area(background_bbox):
                get_background_bboxes(background_bbox, divided_bbox, [bb for bb in bboxes if bbox_intersect(divided_bbox, bb)])
    else:
        if area(handled_bbox) > area(background_bbox):
            background_bbox[:] = handled_bbox

if __name__ == '__main__':
    split = "val"
    dataset = load_dataset("detection-datasets/coco")

    labels = ['background'] + dataset[split].features["objects"]['category'].feature.names
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    for sample in tqdm(dataset[split]):
        image = sample['image']
        image_id = sample['image_id']
        bboxes = sample['objects']['bbox']
        bbox_ids = sample["objects"]["bbox_id"]
        labels = sample["objects"]["category"]

        if image_id == 200365:
            bboxes.pop(2)
            labels.pop(2)
            bbox_ids.pop(2)
        if image_id == 550395:
            bboxes.pop()
            labels.pop()
            bbox_ids.pop()
        if image_id == 158292:
            bboxes.pop(-2)
            labels.pop(-2)
            bbox_ids.pop(-2)
        if image_id == 171360:
            bboxes.pop(-11)
            labels.pop(-11)
            bbox_ids.pop(-11)
        if image_id == 183338:
            bboxes.pop(-1)
            labels.pop(-1)
            bbox_ids.pop(-1)
        if image_id == 340038:
            bboxes.pop(-2)
            labels.pop(-2)
            bbox_ids.pop(-2)

        global background_bbox
        background_bbox = [0, 0, 0, 0]
        get_background_bboxes(background_bbox, [0, 0, image.size[0], image.size[1]], bboxes)
        w_min, h_min = image.size

        for bbox, bbox_id, label in zip(bboxes, bbox_ids, labels):
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            w_min = min(w_min, w)
            h_min = min(h_min, h)
            os.makedirs(f"dataset/{split}/{id2label[str(label + 1)]}", exist_ok=True)
            try:
                image.crop((x1, y1, x2, y2)).save(f"dataset/{split}/{id2label[str(label + 1)]}/{image_id}_{bbox_id}.png")
            except SystemError:
                breakpoint()

        x1, y1, x2, y2 = background_bbox
        os.makedirs(f"dataset/{split}/{id2label['0']}", exist_ok=True)
        w, h = x2 - x1, y2 - y1
        if w > w_min / 2 and h > h_min / 2 and area(background_bbox) > 25:
            try:
                image.crop((x1, y1, x2, y2)).save(f"dataset/{split}/{id2label['0']}/{image_id}_background.png")
            except SystemError:
                breakpoint()