import numpy as np


def cal_norm_vec(point1: np.array, point2: np.array):

    vec = point2-point1
    vec_mag = np.linalg.norm(vec, axis=1)
    vec_norm = np.array(list(map((lambda x, y: x/y), vec, vec_mag)))

    return vec_norm

def cal_local_angle(point1: np.array, point2: np.array, point3: np.array):

    dot_result = np.sum(cal_norm_vec(point1, point2) *
                        cal_norm_vec(point3, point2), axis=1)
    angle = np.rad2deg(np.arccos(dot_result))

    return angle

def find_angle(world_ladk_vid: np.array):

    r_wrist = cal_local_angle(
        world_ladk_vid[:, 20], world_ladk_vid[:, 16], world_ladk_vid[:, 14])
    r_elbow = cal_local_angle(
        world_ladk_vid[:, 16], world_ladk_vid[:, 14], world_ladk_vid[:, 12])
    r_shoulder = cal_local_angle(
        world_ladk_vid[:, 14], world_ladk_vid[:, 12], world_ladk_vid[:, 24])
    r_hip = cal_local_angle(
        world_ladk_vid[:, 12], world_ladk_vid[:, 24], world_ladk_vid[:, 26])
    r_knee = cal_local_angle(
        world_ladk_vid[:, 24], world_ladk_vid[:, 26], world_ladk_vid[:, 28])
    r_ankle = cal_local_angle(
        world_ladk_vid[:, 26], world_ladk_vid[:, 28], world_ladk_vid[:, 32])

    l_wrist = cal_local_angle(
        world_ladk_vid[:, 19], world_ladk_vid[:, 15], world_ladk_vid[:, 13])
    l_elbow = cal_local_angle(
        world_ladk_vid[:, 15], world_ladk_vid[:, 13], world_ladk_vid[:, 11])
    l_shoulder = cal_local_angle(
        world_ladk_vid[:, 13], world_ladk_vid[:, 11], world_ladk_vid[:, 23])
    l_hip = cal_local_angle(
        world_ladk_vid[:, 11], world_ladk_vid[:, 23], world_ladk_vid[:, 25])
    l_knee = cal_local_angle(
        world_ladk_vid[:, 23], world_ladk_vid[:, 25], world_ladk_vid[:, 27])
    l_ankle = cal_local_angle(
        world_ladk_vid[:, 25], world_ladk_vid[:, 27], world_ladk_vid[:, 31])

    return np.array([r_wrist, r_elbow, r_shoulder, r_hip, r_knee, r_ankle,
                    l_wrist, l_elbow, l_shoulder, l_hip, l_knee, l_ankle]).T

def angle_similarity(ang1: np.array, ang2: np.array, weight: np.array, max_angle: int, expo: bool):
    """
    คำนวณความเหมือนกันของมุมในแต่ละข้อพับระหว่าง 2 วิดิโอ (0 - max_angle)

    input:
        ang1 : มุมข้อพับทั้ง 12 มุมของวิดิโอที่ 1 ได้จาก find_angle
        ang2 : มุมข้อพับทั้ง 12 มุมของวิดิโอที่ 2 ได้จาก find_angle
        weight : ค่าถ่วงน้ำหนัก ที่ได้จากฟังก์ชัน gen_weight
        max_angle : ค่าที่ใช้ในการ normalize แต่ละมุมข้อพับ

    output:
        numpy array ความเหมือนของมุมข้อพับ 12 มุมที่คูณกับค่าถ่วงน้ำหนักแล้ว มีช่วงตั้งแต่ 0 - 1
    """

    joint_diff_norm = (max_angle-np.abs(ang1 - ang2)) / max_angle

    if not expo:
        return joint_diff_norm * weight
    else:
        joint_diff_expo = np.where((joint_diff_norm >= 0) & (
            joint_diff_norm <= 1), 1/(1+(joint_diff_norm/(1-joint_diff_norm))**-2), joint_diff_norm)
        # each angle has range between 0 and 1 (default weight)
        return joint_diff_expo * weight

def angle_difference(ang1: np.array, ang2: np.array, weight: np.array, max_angle: int, expo: bool):
    """
    คำนวณความต่างของมุมในแต่ละข้อพับระหว่าง 2 วิดิโอ (0 - max_angle)

    input:
        ang1 : มุมข้อพับทั้ง 12 มุมของวิดิโอที่ 1 ได้จาก find_angle
        ang2 : มุมข้อพับทั้ง 12 มุมของวิดิโอที่ 2 ได้จาก find_angle
        weight : ค่าถ่วงน้ำหนัก ที่ได้จากฟังก์ชัน gen_weight
        max_angle : ค่าที่ใช้ในการ normalize แต่ละมุมข้อพับ

    output:
        numpy array ความต่างของมุมข้อพับ 12 มุมที่คูณกับค่าถ่วงน้ำหนักแล้ว มีช่วงตั้งแต่ 0 - 1
    """

    joint_diff_norm = (np.abs(ang1 - ang2) / max_angle)

    if not expo:
        return joint_diff_norm * weight
    else:
        joint_diff_expo = np.where((joint_diff_norm >= 0) & (
            joint_diff_norm <= 1), 1/(1+(joint_diff_norm/(1-joint_diff_norm))**-2), joint_diff_norm)
        # each angle has range between 0 and 1 (default weight)
        return joint_diff_expo * weight