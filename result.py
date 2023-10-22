import numpy as np
import cv2

def display(path: list, dist_mat: np.array, dist_lndmk_mat: np.array, all_frame1: list, all_frame2: list,
            cam_land_vid1: np.array, cam_land_vid2: np.array):
            
    cost = 0
    for i, j in path:

        img1 = all_frame1[i].copy()
        img2 = all_frame2[j].copy()
        height1, width1, _ = img1.shape
        height2, width2, _ = img2.shape
        height_ratio = cl_height/max(height1, height2)
        width_ratio = cl_width/max(width1, width2)

        add_cl(image=img2,
                y=int((height2/2)-(height2*height_ratio)/2),
                x=0,
                width_ratio=width_ratio,
                height_ratio=height_ratio,
                dist=dist_mat[i, j],
                sim_diff_function=sim_diff_function)

        cost += dist_mat[i, j]/len(path)

        progressBar(image=img2,
                    dist=dist_mat[i, j],
                    cost=cost,
                    sim_diff_function=sim_diff_function)

        for k in range(dist_lndmk_mat[i, j].shape[0]):

            # color_error ใช้ได้กับ similarity อย่างเดียว
            if "similarity" in sim_diff_function.__name__.lower():
                img1 = color_error(
                    cam_land_vid1[i], width1, height1, img1, dist_lndmk_mat[i, j], k, weight, sim_diff_function)
                img2 = color_error(
                    cam_land_vid2[j], width2, height2, img2, dist_lndmk_mat[i, j], k, weight, sim_diff_function)
            elif "difference" in sim_diff_function.__name__.lower():
                img1 = color_error(
                    cam_land_vid1[i], width1, height1, img1, 1-dist_lndmk_mat[i, j], k, weight, sim_diff_function)
                img2 = color_error(
                    cam_land_vid2[j], width2, height2, img2, 1-dist_lndmk_mat[i, j], k, weight, sim_diff_function)

        merge_img = hconcat_resize([img1, img2])

        