import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation

from sys import platform

BUTTONS_DICT = {
    'ARROW_RIGHT': 83,
    'ARROW_LEFT': 81,
    'ARROW_UP': 82,
    'ARROW_DOWN': 84,
    's': 115,
    'q': 113,
    'space': 32,
}

BUTTONS_DICT_MAC = {
    'ARROW_RIGHT': 3,
    'ARROW_LEFT': 2,
    'ARROW_UP': 0,
    'ARROW_DOWN': 1,
    's': 115,
    'q': 113,
    'space': 32,
}

if platform == "darwin":
    BUTTONS_DICT = BUTTONS_DICT_MAC


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class Transformation(object):

    def __init__(self, a, b=None):

        if b is None:
            self.rvec, self.tvec = self.tr4x4_to_rt(a)
            self.tr = a

        else:
            self.rvec = a
            self.tvec = b
            self.tr = self.rt_to_4x4(a, b)

    @staticmethod
    def rt_to_4x4(rvec, tvec):

        tr = np.identity(4)
        tr[0:3, 0:3] = Rotation.from_rotvec(rvec.squeeze()).as_matrix()
        tr[0:3, 3] = tvec.squeeze()

        return tr

    @staticmethod
    def tr4x4_to_rt(tr):
        return Rotation.from_matrix(tr[0:3, 0:3]).as_rotvec(), tr[0:3, 3]

    def __mul__(self, other):

        res = self.tr @ other.tr
        res = Transformation(res)

        return res

    def inverse(self):

        res = Transformation(np.linalg.inv(self.tr))

        return res

import config as cfg

class ObjectToTrack:

    ALPHA = 1
    
    r_outer = cfg.TROCAR_OUTER_R * ALPHA
    r_inner = cfg.TROCAR_INNER_R * ALPHA
    r_bottom = cfg.TROCAR_BOTTOM_R * ALPHA
    height = cfg.TROCAR_HEIGHT * ALPHA
    thickness = cfg.TROCAR_THICKNESS * ALPHA

    original_points = np.array([

        [0, r_outer, height / 2 + thickness],
        [-r_outer, 0, height / 2 + thickness],
        [0, -r_outer, height / 2 + thickness],

        [0, r_outer, height/2],
        [-r_outer, 0, height/2],
        [0, -r_outer, height/2],

        [0, r_bottom, -height/2],
        [r_bottom, 0, -height/2],
        [0, -r_bottom, -height/2]

    ]).astype(np.float32)


def get_cube_grid(h, r):

    return np.array([
        [-r, -r, 0],
        [-r, r, 0],
        [r, r, 0],
        [r, -r, 0],
        [-r, -r, h],
        [-r, r, h],
        [r, r, h],
        [r, -r, h],
    ])


def draw_cube(image, tr, h, r, camera_matrix, dist_coeffs, weight=0.90):

    cube = get_cube_grid(h, r)
    imgpts, _ = cv2.projectPoints(cube, tr.rvec, tr.tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    image_back = image.copy()
    image_back = cv2.drawContours(image_back, [imgpts[:4]], -1, (0, 255, 0, 0), -3)
    for i, j in zip(range(4), range(4, 8)):
        image_back = cv2.line(image_back, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 2)
        image_back = cv2.drawContours(image_back, [imgpts[4:]], -1, (0, 0, 255), 2)

    image = cv2.addWeighted(image, weight, image_back, 1 - weight, 0)

    return image


def get_bbox(segmentation):

    bb_points = np.rint(np.array(np.where(segmentation == 1)).T)
    bbox_min_y = np.min(bb_points[:, 0]) - 10
    bbox_min_x = np.min(bb_points[:, 1]) - 10
    bbox_max_y = np.max(bb_points[:, 0]) + 10
    bbox_max_x = np.max(bb_points[:, 1]) + 10

    bbox = np.array([bbox_min_x, bbox_min_y, bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y]).astype(np.float64)

    return bbox


def read_camera_matrices(path):

    f = open(path)
    calibration_data = json.load(f)
    camera_matrix = np.array(calibration_data["Camera Matrix"])
    dist_coeffs = np.array(calibration_data["Disortion"])

    return camera_matrix, dist_coeffs


def load_np_arrays_from_npz(input_file):

    np.load.__defaults__=(None, True, True, 'ASCII')

    arrays_dict = {}
    loaded_data = np.load(input_file)

    for name in loaded_data.files:
        arrays_dict[name] = loaded_data[name]

    return arrays_dict["arr_0"].item()