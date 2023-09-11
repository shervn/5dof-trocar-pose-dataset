import os
import random

from utils import load_np_arrays_from_npz, NumpyEncoder, read_camera_matrices, Transformation as Tr
import glob
import json
import cv2
import shutil
import config as cfg

data_path = cfg.DATA_PATH
dataset_name = cfg.DATASET_NAME

random.seed(2021)


max_t = 0
min_t = 1000

def convert_to_bop_dataset(poses_dir, output_dir, scene_id=0, stage='train'):

    intrinsics, distortion = read_camera_matrices(f"{poses_dir[0]}/../camera_calibration.json")

    scene_id_str = "{:06d}".format(scene_id)

    output_dir = f"{output_dir}/{stage}_{'primesense' if stage != 'train' else 'pbr'}"

    shutil.rmtree(f"{output_dir}/{scene_id_str}", ignore_errors=True)
    shutil.rmtree(f"{output_dir}/{scene_id_str}/rgb", ignore_errors=True)
    shutil.rmtree(f"{output_dir}/{scene_id_str}/mask", ignore_errors=True)
    shutil.rmtree(f"{output_dir}/{scene_id_str}/mask_visib", ignore_errors=True)

    os.makedirs(f"{output_dir}/{scene_id_str}", exist_ok=True)
    os.makedirs(f"{output_dir}/{scene_id_str}/rgb", exist_ok=True)
    os.makedirs(f"{output_dir}/{scene_id_str}/mask", exist_ok=True)
    os.makedirs(f"{output_dir}/{scene_id_str}/mask_visib", exist_ok=True)

    instance_id = 0

    poses_files = []

    for dir in poses_dir:
        tmp = sorted(glob.glob(f"{dir}/*.npz"))
        poses_files.extend(tmp)

    poses_files.sort()
    count = (int)(len(poses_files) * 0.8)

    if stage == 'train':
        poses_files = random.sample(poses_files, count)
    else:
        poses_files = random.sample(poses_files, len(poses_files) - count)

    image_ids = [i + 1 for i in range(len(poses_files))]

    scene_gt = {}
    scene_camera = {}
    scene_gt_info = {}

    for pose_data_path, image_id in zip(poses_files, image_ids):

        print(pose_data_path)

        pose_data = load_np_arrays_from_npz(pose_data_path)
        image = pose_data["rgb"]
        mask = pose_data["mask"]

        image_name = "{:06d}.png".format(image_id)
        mask_name = "{:06d}_{:06d}.png".format(image_id, 0)

        cv2.imwrite(f"{output_dir}/{scene_id_str}/rgb/{image_name}", image)
        cv2.imwrite(f"{output_dir}/{scene_id_str}/mask/{mask_name}", mask * 255)
        cv2.imwrite(f"{output_dir}/{scene_id_str}/mask_visib/{mask_name}", mask * 255)

        pose = Tr(pose_data["rotation"], pose_data["translation"])
        R = pose.tr[0:3, 0:3]
        t = pose.tr[0:3, 3]

        u = t[2]
        v = t[2]

        global max_t, min_t
        if max_t < u:
            max_t = u

        if min_t > v:
            min_t = v


        scene_gt[str(image_id)] = [
            {
            'obj_id': instance_id,
            'cam_R_m2c': R.flatten().tolist(),
            'cam_t_m2c': t.flatten().tolist(),
            }
        ]

        scene_camera[str(image_id)] = {
            "cam_K": intrinsics.flatten().tolist(),
            "depth_scale": 1,
        }

        scene_gt_info[str(image_id)] = [
            {
                "bbox_obj": pose_data["bbox"],
                "bbox_visib": pose_data["bbox"],
                "px_count_all": pose_data["px_count_all"],
                "px_count_valid": pose_data["px_count_all"],
                "px_count_visib": pose_data["px_count_all"],
                "visib_fract": 1
             }
        ]

    json_object = json.dumps(scene_gt, indent=4,  cls=NumpyEncoder)
    with open(f"{output_dir}/{scene_id_str}/scene_gt.json", "w") as outfile:
        outfile.write(json_object)

    json_object = json.dumps(scene_gt_info, indent=4,  cls=NumpyEncoder)
    with open(f"{output_dir}/{scene_id_str}/scene_gt_info.json", "w") as outfile:
        outfile.write(json_object)

    json_object = json.dumps(scene_camera, indent=4,  cls=NumpyEncoder)
    with open(f"{output_dir}/{scene_id_str}/scene_camera.json", "w") as outfile:
        outfile.write(json_object)


    print(min_t, " ", max_t)


if __name__ == "__main__":

    output_dir = f"{cfg.DATA_PATH}/{cfg.DATASET_OUT_NAME}/bop_dataset/trocar"

    convert_to_bop_dataset([f"{cfg.DATA_PATH}/onetrocar-noneedle/npz"], output_dir, scene_id=0, stage='train')
