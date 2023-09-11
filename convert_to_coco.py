import glob
import cv2
import shutil
import os
import json
import numpy as np
import random

from utils import load_np_arrays_from_npz, NumpyEncoder

import config as cfg
random.seed(2021)

def convert_to_coco(data_paths, stage):

    shutil.rmtree(f"{cfg.DATA_PATH}/{cfg.DATASET_OUT_NAME}/coco/annotations/{stage}", ignore_errors=True)
    shutil.rmtree(f"{cfg.DATA_PATH}/{cfg.DATASET_OUT_NAME}/coco/labels/{stage}", ignore_errors=True)
    shutil.rmtree(f"{cfg.DATA_PATH}/{cfg.DATASET_OUT_NAME}/coco/images/{stage}", ignore_errors=True)

    os.makedirs(f"{cfg.DATA_PATH}/{cfg.DATASET_OUT_NAME}/coco/", exist_ok=True)
    os.makedirs(f"{cfg.DATA_PATH}/{cfg.DATASET_OUT_NAME}/coco/annotations/{stage}", exist_ok=True)
    os.makedirs(f"{cfg.DATA_PATH}/{cfg.DATASET_OUT_NAME}/coco/images/{stage}", exist_ok=True)
    os.makedirs(f"{cfg.DATA_PATH}/{cfg.DATASET_OUT_NAME}/coco/labels/{stage}", exist_ok=True)

    coco_data = {
        "info": {},
        "licenses": [],
        "categories": [{"id": 0, "name": "trocar"}],
        "images": [],
        "nc": 1,
        "names": ["trocar"],
        "train": f"{cfg.DATA_PATH}/{cfg.DATASET_OUT_NAME}/coco/train.txt",
        "val": f"{cfg.DATA_PATH}/{cfg.DATASET_OUT_NAME}/coco/val.txt",
        "annotations": []
    }

    filenames = []

    annotation_id = 1

    all_files = []
    for data_path in data_paths:
        all_files.extend(sorted(glob.glob(f"{data_path}/*.npz")))

    random.shuffle(all_files)

    split_percentage = 0.9
    split_index = int(len(all_files) * split_percentage)

    if stage == 'train':
            all_files = all_files[:split_index]
    else:
        all_files = all_files[split_index:]

    image_id = 0
    for filename in all_files:

        category_id = 1

        npz = load_np_arrays_from_npz(filename)
        image = npz['rgb']

        width, height = image.shape[0], image.shape[1]
        image_path = f"{cfg.DATA_PATH}/{cfg.DATASET_OUT_NAME}/coco/images/{stage}/{image_id}.{cfg.IMAGE_EXTENSION}"
        label_path = f"{cfg.DATA_PATH}/{cfg.DATASET_OUT_NAME}/coco/labels/{stage}/{image_id}.txt"
        # Add image information to COCO data
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_path,
            "width": width,
            "height": height
        })

        cv2.imwrite(image_path, image)
        filenames.append(image_path)

        mask = npz['mask']
        bbox_tmp = npz['bbox'].astype(float)[0:4]

        bbox_tmp[0] += bbox_tmp[2] / 2
        bbox_tmp[1] += bbox_tmp[3] / 2
        bbox_tmp[[0, 2]] /= height
        bbox_tmp[[1, 3]] /= width
        bbox = np.zeros(5)
        bbox[1:] = bbox_tmp
        area = npz['bbox'].astype(float)[2] * npz['bbox'].astype(float)[3]
        bbox_string1 = " ".join(list(map(str, bbox.flatten().tolist())))
        with open(label_path, 'w') as f:
            f.write(bbox_string1)

        mask_coords = np.argwhere(mask == category_id)
        seg_mask = mask_coords.flatten().tolist()

        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [seg_mask],
            "bbox": bbox,
            "area": area,
            "iscrowd": 0  # Set to 1 if dealing with crowd instances
        })

        if len(npz['bbox'].astype(float)) == 8:

            bbox_tmp = npz['bbox'].astype(float)[4:8]
            bbox_tmp[0] += bbox_tmp[2] / 2
            bbox_tmp[1] += bbox_tmp[3] / 2
            bbox_tmp[[0, 2]] /= height
            bbox_tmp[[1, 3]] /= width
            bbox = np.zeros(5)
            bbox[1:] = bbox_tmp
            area = npz['bbox'].astype(float)[6] * npz['bbox'].astype(float)[7]
            bbox_string2 = bbox_string1 + "\n" + " ".join(list(map(str, bbox.flatten().tolist())))
            with open(label_path, 'w') as f:
                f.write(bbox_string2)

            mask_coords = np.argwhere(mask == category_id)
            seg_mask = mask_coords.flatten().tolist()

            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [seg_mask],
                "bbox": bbox,
                "area": area,
                "iscrowd": 0  # Set to 1 if dealing with crowd instances
            })

        annotation_id += 1
        image_id += 1

    output_json = f"{cfg.DATA_PATH}/{cfg.DATASET_OUT_NAME}/coco/annotations/{stage}/output_coco.json"
    with open(output_json, "w") as json_file:
        json.dump(coco_data, json_file, indent=4, cls=NumpyEncoder)

    filenames = '\n'.join(filenames)

    with open(f"{cfg.DATA_PATH}/{cfg.DATASET_OUT_NAME}/coco/{stage}.txt", 'w') as f:
        f.write(filenames)

    txt = f"train: {cfg.DATA_PATH}/{cfg.DATASET_OUT_NAME}/coco/train.txt \n" \
          f"val:  {cfg.DATA_PATH}/{cfg.DATASET_OUT_NAME}/coco/val.txt\n" \
          "\n" \
          "nc: 1 \n" \
          "names: ['trocar'] \n"

    with open(f"{cfg.DATA_PATH}/{cfg.DATASET_OUT_NAME}/coco/coco.yml", 'w') as f:
        f.write(txt)

    print(f"COCO data saved to {output_json}")


if __name__ == "__main__":

    poses_dir = [
                    f"{cfg.DATA_PATH}/validation/npz",
                    f"{cfg.DATA_PATH}/onetrocar-needle/npz",
                    f"{cfg.DATA_PATH}/onetrocar-noneedle/npz",
    ]

    convert_to_coco(poses_dir, "train")
    convert_to_coco(poses_dir, "val")