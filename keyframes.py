def match_kf(keyframe_list, dist_mat):
    student_kf = [0]

    for idx, kf in enumerate(keyframe_list):
        match_idx = (np.where(dist_mat[kf, student_kf[idx]:] >= (np.max(dist_mat[kf, student_kf[idx]:]))))[0][0]
        student_kf.append(student_kf[idx]+match_idx)
    student_kf.append(dist_mat.shape[1])

    keyframe_list.insert(0,0)
    keyframe_list.append(dist_mat.shape[0])
    return keyframe_list, student_kf