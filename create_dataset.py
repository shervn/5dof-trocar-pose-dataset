import glob
import cv2
import numpy as np
import trimesh
import pyrender

from utils import Transformation as Tr, ObjectToTrack
from utils import read_camera_matrices, get_bbox, BUTTONS_DICT

import config as cfg

refinement_step = 0.1
refine_x_or_y = 0
refinement_vector = [0, 0, 0]
rotation_refinement = [0, 0, 0]

refinement_x_step = 0.02
refinement_y_step = 0.02
refinement_z_step = 0.02


def solve_initial_pnp(frame_path, object, camera_matrix, dist_coeffs, initial_points=[]):

    img = None
    global m_x, m_y
    m_x = None
    m_y = None

    def draw_circle(event, x, y, flags, param):

        global m_x, m_y

        if event == cv2.EVENT_LBUTTONDOWN:
            if m_x is None:
                m_x, m_y = y, x
            else:
                mouseX, mouseY = x + m_y - 50, y + m_x - 50
                cv2.circle(img, (mouseX, mouseY), 1, (255, 0, 0), 1)
                initial_points.append([mouseX, mouseY])

    img = read_image(frame_path)

    if not initial_points:

        cv2.namedWindow('Get Neighborhood')
        cv2.setMouseCallback('Get Neighborhood', draw_circle)
        cv2.imshow('Get Neighborhood', img)

        cv2.waitKey(0)
        cv2.destroyWindow('Get Neighborhood')

        cv2.namedWindow('PnP Resolver', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("PnP Resolver", 800, 800)
        cv2.setMouseCallback('PnP Resolver', draw_circle)

        while len(initial_points) != object.original_points.shape[0]:
            tmp = img[m_x - 50:m_x + 50, m_y - 50:m_y + 50].astype(float).copy()
            tmp -= tmp.min()
            tmp /= tmp.max()
            tmp *= 255
            tmp = tmp.astype(np.uint8)
            cv2.imshow('PnP Resolver', tmp)
            cv2.waitKey(0)

        cv2.destroyWindow('PnP Resolver')

    initial_points = np.array(initial_points).astype(float)
    _, r_camera_object, t_camera_object = cv2.solvePnP(object.original_points, initial_points,
                                                       camera_matrix, dist_coeffs,
                                                       None, None, False, cv2.SOLVEPNP_EPNP)

    print("*** Selected Points ***")
    print(initial_points)
    print("**********************")
    print("***** PnP Result *****")
    print("Translation: \n", t_camera_object)
    print("Rotation: \n", r_camera_object)
    print("**********************")

    return Tr(r_camera_object, t_camera_object)


def estimate_camera_board(img, aruco_dictionary, aruco_grid, camera_matrix, dist_coeffs):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        gray, dictionary=aruco_dictionary)

    print(len(rejectedImgPoints))

    # cv2.aruco.drawDetectedMarkers(img, corners, ids)
    _, r_camera_board, t_camera_board = cv2.aruco.estimatePoseBoard(corners, ids, aruco_grid,
                                                                    camera_matrix, dist_coeffs, None, None)
    return Tr(r_camera_board, t_camera_board)


def refine_with_translation(translation_refined, tr_camera_object, tr_camera_board):

    translation_refined = np.array(translation_refined)
    tr_object_objectrefined = Tr(np.array(rotation_refinement), np.array(translation_refined).astype(float))

    tr_object_board = tr_camera_object.inverse() * tr_camera_board
    tr_objectrefined_board = tr_object_objectrefined.inverse() * tr_object_board

    return tr_objectrefined_board


def estimate_object_board(frame_path, object, camera_matrix, dist_coeffs, aruco_dictionary, aruco_board, initial_points=[]):

    img = read_image(frame_path)

    tr_camera_object_pnp = solve_initial_pnp(frame_path=frame_path,
                                             object=object,
                                             camera_matrix=camera_matrix,
                                             dist_coeffs=dist_coeffs,
                                             initial_points=initial_points)

    tr_camera_board = estimate_camera_board(img=img,
                                            aruco_dictionary=aruco_dictionary,
                                            aruco_grid=aruco_board,
                                            camera_matrix=camera_matrix,
                                            dist_coeffs=dist_coeffs)

    tr_object_board_refined = refine_with_translation([0, 0, 0], tr_camera_object_pnp, tr_camera_board)

    return tr_object_board_refined


def get_rendered_object(tr_object_opencv, mesh, camera, original_image):

    camera = pyrender.IntrinsicsCamera(fx=camera[0, 0],
                                       fy=camera[1, 1],
                                       cx=camera[0, 2],
                                       cy=camera[1, 2],
                                       znear=0.001, zfar=10000)
    scene = pyrender.Scene(ambient_light=(1, 1, 1))

    camera_pose = tr_object_opencv.tr.copy()
    camera_pose[0:3, 3] = camera_pose[0:3, 3] / 1000
    camera_pose[[1, 2]] *= -1

    scene.add(camera)
    scene.add(mesh, "mesh",  pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=original_image.shape[1],
                                          viewport_height=original_image.shape[0])
    color, _ = renderer.render(scene)

    rendered_image = color.astype(np.uint8)

    alpha = 0.30  # Adjust the blending factor as needed
    output_image = cv2.addWeighted(original_image, 1 - alpha, rendered_image, alpha, 0)

    segmentation = color[..., 0].copy()
    segmentation[segmentation < 255] = 1
    segmentation[segmentation != 1] = 0

    return output_image, segmentation


def get_segmentation_and_render_from_path(img, tr_objectBoard, refinement_vector, camera_matrix, dist_coeffs):

    tr_cameraBoard = estimate_camera_board(img=img,
                                           aruco_dictionary=aruco_dictionary,
                                           aruco_grid=aruco_board,
                                           camera_matrix=camera_matrix,
                                           dist_coeffs=dist_coeffs)

    tr_cameraObject = tr_cameraBoard * tr_objectBoard.inverse()
    tr_objectBoard_refined = refine_with_translation(refinement_vector, tr_cameraObject, tr_cameraBoard)
    tr_cameraObject = tr_cameraBoard * tr_objectBoard_refined.inverse()

    rendered_image, segmentation = get_rendered_object(tr_cameraObject, trocar_mesh, camera_matrix, img)
    segmentation_vis = rendered_image.copy()

    rendered_with_axis = cv2.drawFrameAxes(rendered_image.copy(), camera_matrix, dist_coeffs,
                                           tr_cameraObject.rvec,
                                           tr_cameraObject.tvec,
                                           ObjectToTrack.height * 4, 5)

    segmentation_vis[segmentation == 1] = [255, 255, 255]

    return img, tr_cameraObject, rendered_image, segmentation, rendered_with_axis, segmentation_vis


CAMERA_MATRIX, DIST_COEFS = read_camera_matrices(
        f"{cfg.DATA_PATH}/{cfg.DATASET_NAME}/camera_calibration.json")


def read_image(path):

    img1 = cv2.imread(path)
    img1 = cv2.resize(img1, (1280, 960))
    img1 = cv2.undistort(img1, CAMERA_MATRIX, DIST_COEFS, None)

    return img1


if __name__ == "__main__":

    trocar_mesh = pyrender.Mesh.from_trimesh(trimesh.load(cfg.CAD_PATH))

    first_frame_name = cfg.PNP_IMG

    camera_matrix, _ = read_camera_matrices(
        f"{cfg.DATA_PATH}/{cfg.DATASET_NAME}/camera_calibration.json")
    dist_coeffs = np.zeros_like(DIST_COEFS)

    aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_1000)
    aruco_board = cv2.aruco.GridBoard.create(markersX=cfg.ARUCO_MARKERS_X,
                                             markersY=cfg.ARUCO_MARKERS_Y,
                                             markerLength=cfg.ARUCO_MARKERS_LENGTH,
                                             markerSeparation=cfg.ARUCO_MARKERS_SEPARATION,
                                             dictionary=aruco_dictionary)

    initial_points = []

    tr_objectBoard = estimate_object_board(f"{cfg.DATA_PATH}/{cfg.DATASET_NAME}/rgb/
                                           {first_frame_name}.{cfg.IMAGE_EXTENSION}",
                                           ObjectToTrack,
                                           camera_matrix, dist_coeffs,
                                           aruco_dictionary, aruco_board,
                                           initial_points=initial_points)

    images = glob.glob(f"{cfg.DATA_PATH}/{cfg.DATASET_NAME}/rgb/*.{cfg.IMAGE_EXTENSION}")
    images.sort()

    f1_path = f"{cfg.DATA_PATH}/{cfg.DATASET_NAME}/rgb/{cfg.img_1}.png"
    f2_path = f"{cfg.DATA_PATH}/{cfg.DATASET_NAME}/rgb/{cfg.img_2}.png"
    f3_path = f"{cfg.DATA_PATH}/{cfg.DATASET_NAME}/rgb/{cfg.img_3}.png"

    img1 = read_image(f1_path)
    img2 = read_image(f2_path)
    img3 = read_image(f3_path)

    k = 0
    while True:

        image1_extracted = get_segmentation_and_render_from_path(img1, tr_objectBoard, refinement_vector,
                                                                 camera_matrix, dist_coeffs)
        image2_extracted = get_segmentation_and_render_from_path(img2, tr_objectBoard, refinement_vector,
                                                                 camera_matrix, dist_coeffs)
        image3_extracted = get_segmentation_and_render_from_path(img3, tr_objectBoard, refinement_vector,
                                                                 camera_matrix, dist_coeffs)

        renders = np.concatenate((image1_extracted[-2], image2_extracted[-2], image3_extracted[-2]), axis=1)
        segmentations = np.concatenate((image1_extracted[-1], image2_extracted[-1], image3_extracted[-1]), axis=1)
        all = np.concatenate((renders, segmentations), axis=0)
        all = cv2.resize(all, (0, 0), fx=0.7, fy=0.7)

        cv2.imshow("Refine", all)
        k = cv2.waitKey(0)

        if k == BUTTONS_DICT['ARROW_RIGHT']:
            refinement_vector[refine_x_or_y] += refinement_step

        if k == BUTTONS_DICT['ARROW_LEFT']:
            refinement_vector[refine_x_or_y] -= refinement_step

        if k == BUTTONS_DICT['ARROW_UP']:
            refinement_vector[2] += refinement_step

        if k == BUTTONS_DICT['ARROW_DOWN']:
            refinement_vector[2] -= refinement_step

        if k == BUTTONS_DICT['s']:
            refine_x_or_y = 1 - refine_x_or_y

        if k == ord('1'):
            refinement_x_step *= -1

        if k == ord('2'):
            refinement_y_step *= -1

        if k == ord('3'):
            refinement_z_step *= -1

        if k == ord('x'):
            rotation_refinement[0] += refinement_x_step

        if k == ord('y'):
            rotation_refinement[1] += refinement_y_step

        if k == ord('z'):
            rotation_refinement[2] += refinement_z_step

        if k == BUTTONS_DICT['q']:
            break

        if k == BUTTONS_DICT['space']:
            break

        if k == BUTTONS_DICT['space']:
            cv2.destroyWindow('Refine')
            break

    print("Saving....")

    count = 0
    for fname in images[:1000]:
        print(fname)
        try:

            img = read_image(fname)

            image_extracted = get_segmentation_and_render_from_path(img,
                                                                    tr_objectBoard,
                                                                    refinement_vector,
                                                                    camera_matrix,
                                                                    dist_coeffs)

            bbox = get_bbox(image_extracted[3])

            cv2.imwrite(f"{cfg.DATA_PATH}/{cfg.DATASET_NAME}/gt/{count}.png", image_extracted[2])

            np.savez(f"{cfg.DATA_PATH}/{cfg.DATASET_NAME}/npz/{count}.npz", {
                "rotation": image_extracted[1].rvec,
                "translation": image_extracted[1].tvec,
                "image_path": fname,
                "bbox": bbox,
                "px_count_all": bbox[2] * bbox[3],
                "rgb": image_extracted[0],
                "mask": image_extracted[3],
                "gt_rgb": image_extracted[4],
                "gt_rendererd": image_extracted[2]
            })
            count += 1

        except:
            print("failed ")

