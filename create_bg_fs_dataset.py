import os

from datasets import load_dataset
from fsdetection import load_fs_dataset
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
        divided_boxes.append([x1, y4, x2, y2])  # Bottom part
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
                get_background_bboxes(background_bbox, divided_bbox,
                                      [bb for bb in bboxes if bbox_intersect(divided_bbox, bb)])
    else:
        if area(handled_bbox) > area(background_bbox):
            background_bbox[:] = handled_bbox


def main(dataset, split, shot, seed):
    dataset = load_fs_dataset(f"HichTala/{dataset}")
    dataset[split].sampling(shots=shot, seed=seed)

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

        do_break = False
        for bbox, bbox_id, label in zip(bboxes, bbox_ids, labels):
            if os.path.exists(f"dataset/{split}/{id2label[str(label + 1)]}/{image_id}_{bbox_id}.png"):
                do_break = True
                break
        if do_break:
            do_break = False
            continue

        new_format = []
        for bbox in bboxes:
            x1, y1, w, h = bbox
            x2, y2 = w + x1, h + y1
            new_format.append((x1, y1, x2, y2))
        bboxes = new_format

        global background_bbox
        background_bbox = [0, 0, 0, 0]
        if len(bboxes) <= 250:
            get_background_bboxes(background_bbox, [0, 0, image.size[0], image.size[1]], bboxes)
        w_min, h_min = image.size

        for bbox, bbox_id, label in zip(bboxes, bbox_ids, labels):
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            w_min = min(w_min, w)
            h_min = min(h_min, h)
            os.makedirs(f"dataset/{split}/{id2label[str(label + 1)]}", exist_ok=True)
            try:
                image.crop((x1, y1, x2, y2)).save(
                    f"dataset/{split}/{id2label[str(label + 1)]}/{image_id}_{bbox_id}.png")
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


if __name__ == '__main__':
    splits = ["train"]
    shots = [1, 5, 10]
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    datasets = ["dota", "dior", "xview", "oktoberfest", "fashionpedia", "cadot", "artaxor", "deepfruits", "uodd"]

    for dataset_name in datasets:
        for shot in shots:
            for seed in seeds:
                for split in splits:
                    main(dataset_name, split, shot, seed)
                dataset = load_dataset("imagefolder", data_dir="dataset", drop_labels=False)
                dataset.push_to_hub(f"HichTala/{dataset_name}_{shot}shot_{seed}")
                os.system("rm -rf dataset")