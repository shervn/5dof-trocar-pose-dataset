import glob
import os
import random
import cv2
import numpy as np
import trimesh
import pyrender
import shutil

from utils import Transformation as Tr, ObjectToTrack
from utils import read_camera_matrices, draw_cube, get_bbox
from scipy.spatial.transform import Rotation as R

import config as cfg

voc_images = glob.glob(f"{cfg.VOC_PATH}/*.jpg")


def get_random_tr():

    rotation = R.random().as_rotvec()
    translation_x = random.uniform(-20, 20)
    translation_y = random.uniform(-20, 20)
    translation_z = random.uniform(55, 100)

    return Tr(rotation, np.array((translation_x, translation_y, translation_z)))


def get_rendered_object(tr_object_opencv, mesh, camera):

    original_image = cv2.imread(random.choice(voc_images))
    original_image = cv2.resize(original_image, (1280, 960))

    camera = pyrender.IntrinsicsCamera(fx=camera[0, 0], fy=camera[1, 1],
                                       cx=camera[0, 2], cy=camera[1, 2],
                                       znear=0.001, zfar=10000)
    scene = pyrender.Scene()
    m = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(m)

    camera_pose = tr_object_opencv.tr.copy()
    camera_pose[[1, 2]] *= -1
    camera_pose[0:3, 3] = camera_pose[0:3, 3] / 1000

    scene.add(camera)
    scene.add(mesh, "mesh",  pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=original_image.shape[1],
                                          viewport_height=original_image.shape[0])
    color, _ = renderer.render(scene)

    rendered_image = color.astype(np.uint8)
    segmentation = color[..., 0].copy()
    segmentation[segmentation < 254] = 1
    segmentation[segmentation != 1] = 0

    segmentation_stacked = np.stack([segmentation, segmentation, segmentation], axis=-1)

    original_image = original_image * (1-segmentation_stacked) + segmentation_stacked * rendered_image

    return original_image, segmentation


if __name__ == "__main__":

    data_count = 400
    trocar_mesh = pyrender.Mesh.from_trimesh(trimesh.load(cfg.CAD_PATH))

    camera_matrix, dist_coeffs = read_camera_matrices(
        f"{cfg.DATA_PATH}/{cfg.DATASET_NAME}/camera_calibration.json")

    shutil.rmtree(f"{cfg.DATA_PATH}/{cfg.DATASET_NAME}/npz", ignore_errors=True)
    os.makedirs(f"{cfg.DATA_PATH}/{cfg.DATASET_NAME}/npz", exist_ok=True)

    count = 0
    while count < 1000:

        try:
            random_tr_cameraObject = get_random_tr()
            img, segmentation = get_rendered_object(random_tr_cameraObject, trocar_mesh, camera_matrix)
            img_raw = img.copy()

            colorFrame = cv2.drawFrameAxes(img, camera_matrix, dist_coeffs,
                                           random_tr_cameraObject.rvec, random_tr_cameraObject.tvec, ObjectToTrack.height * 4, 2)

            colorFrame = draw_cube(colorFrame, random_tr_cameraObject,
                                   ObjectToTrack.height, ObjectToTrack.r_outer,
                                   camera_matrix, dist_coeffs, 0.70)

            bbox = get_bbox(segmentation)

            np.savez(f"{cfg.DATA_PATH}/{cfg.DATASET_NAME}/npz/{count}.npz", {
                "rotation": random_tr_cameraObject.rvec,
                "translation": random_tr_cameraObject.tvec,
                "bbox": bbox,
                "px_count_all": bbox[2] * bbox[3],
                "rgb": img_raw,
                "mask": segmentation,
                "gt_rgb": colorFrame,
            })

            count += 1
        except:
            continue
