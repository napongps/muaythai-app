import numpy as np
import cv2
import os
from angle import *
from cosine import *
from concurrent.futures import ProcessPoolExecutor

def form_vec(coor_arr: np.array):
    """
    เอาเฉพาะ x กับ y ในตำแหน่งข้อต่อ

    input:
        coor_arr: ตำแหน่งข้อต่อ 3 มิติ (x,y,z)

    output:
        ตำแหน่งข้อต่อ 2 มิติ
    """
    return (int(coor_arr[0]), int(coor_arr[1]))

def denormalize_landmark(cam_landmark: np.array, mapped_landmark: list, width: int, height: int):
    """
    denormalize landmark

    input:
        cam_landmark : ตำแหน่งข้อต่อแบบ camera coordinate
        mapped_landmark : ตำแหน่ง landmark ที่ k ใน cam_landmark
        width : ความกว้างของรูป (x)
        height : ความสูงของรูป (y)

    output:
        denorm_landmark : denormalized landmark
    """

    denorm_landmark = []
    for i in mapped_landmark:
        denorm_landmark.append(
            [cam_landmark[i][0]*width, cam_landmark[i][1]*height, cam_landmark[i][2]*width])
    denorm_landmark.append(
        [cam_landmark[19][0]*width, cam_landmark[19][1]*height, cam_landmark[19][2]*width])
    denorm_landmark.append(
        [cam_landmark[20][0]*width, cam_landmark[20][1]*height, cam_landmark[20][2]*width])
    denorm_landmark.append(
        [cam_landmark[31][0]*width, cam_landmark[31][1]*height, cam_landmark[31][2]*width])
    denorm_landmark.append(
        [cam_landmark[32][0]*width, cam_landmark[32][1]*height, cam_landmark[32][2]*width])

    return denorm_landmark

def map_landmark(sim_diff_function, k: int):
    """
    ฟังก์ชัน map ตำแหน่งของข้อพับทั้ง 12 หรือท่อนแขนท่อนขาทั้ง 16 เป็นตำแหน่งใน cam_landmark

    input:
        sim_diff_function : ฟังก์ชันที่ใช้ในการคำนวณค่าคะแนน
        k : ข้อพับ/ท่อนแขนท่อนขาที่เท่าไหร่

    output:
        dict_map[k] : ตำแหน่งของข้อพับ/ท่อนแขนท่อนขาที่ k ใน cam_landmark
    """

    if "angle" in sim_diff_function:
        dict_map = {0: [16], 1: [14], 2: [12], 3: [24], 4: [26], 5: [
            28], 6: [15], 7: [13], 8: [11], 9: [23], 10: [25], 11: [27]}
    elif "cosine" in sim_diff_function:
        dict_map = {0: [28, 32], 1: [26, 28], 2: [24, 26], 3: [12, 24], 4: [16, 20], 5: [14, 16], 6: [12, 14], 7: [27, 31], 8: [25, 27], 9: [23, 25],
                    10: [11, 23], 11: [15, 19], 12: [13, 15], 13: [11, 13], 14: [11, 12], 15: [23, 24]}
    return dict_map[k]

def color_error(cam_landmark: np.array, width: int, height: int, image: np.array, dist_lndmk_mat: np.array, k: int, 
            sim_diff_function):

    """
    ฟังก์ชันแสดงจุดที่ผิดพลาด

    input:
        cam_landmark : ตำแหน่งข้อต่อแบบ camera coordinate
        width : ความกว้างของรูป (x)
        height : ความสูงของรูป (y)
        image : รูปภาพ frame ที่ต้องการ
        dist_lndmk_mat : ค่าคะแนนความต่าง/ความเหมือน ของแต่ละข้อต่อ ของทุกคู่เฟรม (แยกข้อต่อ/แยกท่อนแขนท่อนขา)
        k : ข้อต่อ/ท่อนแขนท่อนขา ที่เท่าไหร่ใน dist_lndmk_mat
        sim_diff_function : ฟังก์ชันที่ใช้คำนวณความต่าง/ความเหมือน

    output:
        new_image : ภาพที่แสดงจุดผิดพลาดไป 1 ข้อต่อ/ท่อนแขนท่อนขา
    """

    annotated_img = image.copy()
    denorm_landmark = denormalize_landmark(
        cam_landmark, map_landmark(sim_diff_function, k), width, height)

    for i in range(len(denorm_landmark)):
        cv2.circle(image, (int(denorm_landmark[i][0]), int(
            denorm_landmark[i][1])), 4, (255, 255, 255), -1)

    ori_similarity = dist_lndmk_mat[k]

    if "angle" in sim_diff_function:
        bright = ori_similarity*255
        size = int(max(5, min(20, 1/(ori_similarity))))

        cv2.circle(annotated_img, (int(denorm_landmark[0][0]), int(
            denorm_landmark[0][1])), size, (0, bright, 255), -1)

    elif "cosine" in sim_diff_function:
        bright = ori_similarity*255
        size = int(max(3, min(20, 1/(ori_similarity))))

        cv2.line(annotated_img, form_vec(denorm_landmark[0]), form_vec(
            denorm_landmark[1]), (0, bright, 255), size)

    opacity = 0.4
    new_image = cv2.addWeighted(annotated_img, opacity, image, 1 - opacity, 0)

    return new_image

def hconcat_resize(img_list: list):
    """
    ต่อภาพที่ขนาดไม่เท่ากัน

    input:
        img_list : list ของรูปภาพที่ต้องการต่อ ex. [img1,img2]

    output:
        im_list_resize : รูปภาพที่ต่อกันแล้ว
    """
    # take minimum hights
    h_min = min(img.shape[0]
                for img in img_list)

    # image resizing
    im_list_resize = [cv2.resize(img,
                                 (int(img.shape[1] * h_min / img.shape[0]),
                                  h_min), interpolation=cv2.INTER_LINEAR)
                      for img in img_list]

    # return final image
    return cv2.hconcat(im_list_resize)

def display_error(path: list, dist_mat: np.array, dist_lndmk_mat: np.array, all_frame1: list, all_frame2: list,
            cam_land_vid1: np.array, cam_land_vid2: np.array, sim_diff_function):

    merge_img = []
    cost = 0

    for i, j in path:

        img1 = all_frame1[i].copy()
        img2 = all_frame2[j].copy()
        height1, width1, _ = img1.shape
        height2, width2, _ = img2.shape

        cost += dist_mat[i, j]/len(path)

        for k in range(dist_lndmk_mat[i, j].shape[0]):

            # color_error ใช้ได้กับ similarity อย่างเดียว
            if "similarity" in sim_diff_function:
                img1 = color_error(
                    cam_land_vid1[i], width1, height1, img1, dist_lndmk_mat[i, j], k, sim_diff_function)
                img2 = color_error(
                    cam_land_vid2[j], width2, height2, img2, dist_lndmk_mat[i, j], k, sim_diff_function)
            elif "difference" in sim_diff_function:
                img1 = color_error(
                    cam_land_vid1[i], width1, height1, img1, 1-dist_lndmk_mat[i, j], k, sim_diff_function)
                img2 = color_error(
                    cam_land_vid2[j], width2, height2, img2, 1-dist_lndmk_mat[i, j], k, sim_diff_function)
    
        merge_img.append(hconcat_resize([img1, img2]))


    return merge_img    
        