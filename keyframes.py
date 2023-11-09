from copy import copy
import numpy as np

def match_kf(keyframe_list, dist_mat):

    keyframe_copy = copy(keyframe_list)
    student_kf = []
    kf_score = []
    penalty = np.arange(0,0.3,0.1)

    for p in penalty:
        student_kf_temp = [0]
        kf_score_temp = []

        # เริ่มต้น
        low_bound = np.max(dist_mat[keyframe_copy[0], 0:])-p
        match_idx = np.where(dist_mat[keyframe_copy[0], 0:] >= low_bound)[0] #ครั้งแรกจะมีแค่ max อันเดียว, มากกว่า max ไม่มี
        min_match_idx = np.where(dist_mat[keyframe_copy[0], 0:] == np.sort(dist_mat[keyframe_copy[0], match_idx])[0])[0][0]

        student_kf_temp.append(min_match_idx)
        kf_score_temp.append(dist_mat[keyframe_copy[0], min_match_idx])

        for idx, kf in enumerate(keyframe_copy[1:]):
            idx = idx+1
            if keyframe_copy[-1] != kf:
                # ไส้กลาง
                low_bound = np.max(dist_mat[kf, student_kf_temp[idx]:])-p
                match_idx = student_kf_temp[idx]+np.where(dist_mat[kf, student_kf_temp[idx]:] >= low_bound)[0]
                min_match_idx = student_kf_temp[idx]+np.where(dist_mat[kf, student_kf_temp[idx]:] == np.sort(dist_mat[kf, match_idx])[0])[0][0]

                student_kf_temp.append(min_match_idx)
                kf_score_temp.append(dist_mat[kf, min_match_idx])

            else:
                # ตอนสุดท้าย

                low_bound = np.max(dist_mat[kf, student_kf_temp[idx]:])
                match_idx = student_kf_temp[idx]+np.where(dist_mat[kf, student_kf_temp[idx]:] >= low_bound)[0]
                min_match_idx = student_kf_temp[idx]+np.where(dist_mat[kf, student_kf_temp[idx]:] == np.sort(dist_mat[kf, match_idx])[-1])[0][0]

                student_kf_temp.append(min_match_idx)
                kf_score_temp.append(dist_mat[kf, min_match_idx])

        kf_score.append(np.mean(kf_score_temp))
        student_kf.append(student_kf_temp)

    keyframe_copy.insert(0,0)
    result =  student_kf[np.argmax(kf_score)]
    if dist_mat.shape[0]-1 not in keyframe_copy:
        keyframe_copy.append(dist_mat.shape[0]-1)
        result.append(dist_mat.shape[1]-1)

    if len(student_kf[np.argmax(kf_score)]) != len(keyframe_copy):
        print(keyframe_copy)
        print(student_kf[np.argmax(kf_score)])

    return keyframe_copy, result