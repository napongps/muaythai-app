import os
import math
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import pickle
from glob import glob


mp_pose = mp.solutions.pose

pose_video = mp_pose.Pose(static_image_mode=False,
                          min_detection_confidence=0.5, model_complexity=2)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def detectPose(frame: np.array):
    """
    Input:
        frame = numpy array of frame that is extracted by cv2

    Output:
        output_image = image that has skeleton in that frame
        norm_landmarks = camera coordinate every keypoint in that frame
        norm_world_landmarks = world coordinate every keypoint in that frame
    """

    output_image = frame.copy()
    output_image.flags.writeable = False
    result = pose_video.process(output_image)

    output_image.flags.writeable = True
    if result.pose_landmarks:
        norm_landmarks = result.pose_landmarks.landmark  # camera coordinate
        norm_world_landmarks = result.pose_world_landmarks.landmark  # world coordinate

        # เอา landmark ที่ไม่ได้ใช้ออก ใบหน้า นิ้วต่างๆ
        exclude_lndmk = list(range(11))
        exclude_lndmk.extend([29, 30, 17, 18, 21, 22])

        for i in exclude_lndmk:
            norm_landmarks[i].visibility = 0

        mp_drawing.draw_landmarks(image=output_image,
                                  landmark_list=result.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    return output_image, norm_landmarks, norm_world_landmarks


def formatJoint(landmarks_obj: list):
    """
    Input:
        landmarks_obj = result landmarks that get from mediapipe

    Output:
        list of x,z,y coordinate for every keypoint eg. [[x1, y1, z1], [x2, y1, z1], ..., [x33, y33, z33]]
    """

    return list(map(lambda ladk: [ladk.x, ladk.y, ladk.z], landmarks_obj))


def landmark_detection(path_vid: str):
    """
    Input:
        path_vid = path of video

    Output:
        cam_ladk = numpy array that contain camera coordinate of all keypoint in every frame of the video
        world_ladk = numpy array that contain world coordinate of all keypoint in every frame of the video
        all_frame = list that contain all skeletal frame in the video
    """

    video = cv2.VideoCapture(path_vid)

    all_frame = []
    cam_ladk = []
    world_ladk = []
    count = 0

    while video.isOpened():

        ret, frame = video.read()

        if not ret:
            break

        # กรณีที่ mediapipe ไม่สามารถหา landmark ได้ detectPose จะเป็น Nonetype
        try:
            output_image, norm_landmarks, norm_world_landmarks = detectPose(
                frame)
        except:
            continue

        all_frame.append(output_image)
        cam_ladk.append(formatJoint(norm_landmarks))
        world_ladk.append(formatJoint(norm_world_landmarks))

        count += 1

    video.release()
    cv2.destroyAllWindows()
    return np.array(cam_ladk), np.array(world_ladk), all_frame
