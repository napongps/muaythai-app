import numpy as np
from angle import *
from cosine import *

def diff_move(extracted_ladk: np.array, windows: int, curr_frame: int, norm_value: int):

    if extracted_ladk.shape[1] == 12:
        adj_diff = angle_difference(ang1=extracted_ladk[curr_frame],
                                    ang2=extracted_ladk[curr_frame-1],
                                    weight=np.ones(12),
                                    max_angle=norm_value,
                                    expo=False)

    if extracted_ladk.shape[1] == 16:
        adj_diff = cosine_difference(vec1=extracted_ladk[curr_frame],
                                        vec2=extracted_ladk[curr_frame-1],
                                        weight=np.ones(16),
                                        max_cosine_diff=norm_value,
                                        expo=False)

    return adj_diff


def MA_W(extracted_ladk: np.array, windows: int, norm_value: int):

    frame = extracted_ladk.shape[0]

    if (windows+windows) > frame:
        frame_diff = (windows+windows) - frame
        extracted_ladk = np.insert(
            extracted_ladk, 0, [extracted_ladk[0]]*(math.ceil(frame_diff/2)+1), axis=0)
        extracted_ladk = np.insert(
            extracted_ladk, -1, [extracted_ladk[-1]]*(math.ceil(frame_diff/2)+1), axis=0)
        frame = extracted_ladk.shape[0]

    all_adj_diff = []
    for curr_frame in range(1, frame):
        all_adj_diff.append(diff_move(extracted_ladk=extracted_ladk,
                                    windows=windows,
                                    curr_frame=curr_frame,
                                    norm_value=norm_value))

    all_adj_diff.append(all_adj_diff[-1])
    all_adj_diff = np.array(all_adj_diff)

    # sliding window
    all_weight = []
    for curr_frame in range(windows, frame-windows+1):
        forward_dist = np.sum(
            all_adj_diff[curr_frame:curr_frame+windows+1], axis=0)
        backward_dist = np.sum(
            all_adj_diff[curr_frame-windows:curr_frame], axis=0)
        total_dist = (forward_dist+backward_dist)
        all_weight.append(total_dist)

    all_weight = np.array(all_weight)
    all_weight = (all_weight/windows)+1

    # add first n frame to all_weight
    all_weight = np.insert(all_weight, 0, [all_weight[0]]*windows, axis=0)

    # add last n frame to all_weight
    all_weight = np.insert(all_weight, -1, [all_weight[-1]]*windows, axis=0)

    return all_weight