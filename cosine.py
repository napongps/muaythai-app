import numpy as np

def find_limb(world_ladk_vid: np.array):
    """
    Output:
        result[limb][frame][coordinate]
    """

    R_findex_ankle = world_ladk_vid[:, 32] - world_ladk_vid[:, 28]
    R_ankle_knee = world_ladk_vid[:, 28] - world_ladk_vid[:, 26]
    R_knee_hip = world_ladk_vid[:, 26] - world_ladk_vid[:, 24]
    R_hip_shoulder = world_ladk_vid[:, 24] - world_ladk_vid[:, 12]
    R_index_wrist = world_ladk_vid[:, 20] - world_ladk_vid[:, 16]
    R_wrist_elbow = world_ladk_vid[:, 16] - world_ladk_vid[:, 14]
    R_elbow_shoulder = world_ladk_vid[:, 14] - world_ladk_vid[:, 12]

    L_findex_ankle = world_ladk_vid[:, 31] - world_ladk_vid[:, 27]
    L_ankle_knee = world_ladk_vid[:, 27] - world_ladk_vid[:, 25]
    L_knee_hip = world_ladk_vid[:, 25] - world_ladk_vid[:, 23]
    L_hip_shoulder = world_ladk_vid[:, 23] - world_ladk_vid[:, 11]
    L_index_wrist = world_ladk_vid[:, 19] - world_ladk_vid[:, 15]
    L_wrist_elbow = world_ladk_vid[:, 15] - world_ladk_vid[:, 13]
    L_elbow_shoulder = world_ladk_vid[:, 13] - world_ladk_vid[:, 11]

    lshoulder_rshoulder = world_ladk_vid[:, 11] - world_ladk_vid[:, 12]
    lhip_rhip = world_ladk_vid[:, 23] - world_ladk_vid[:, 24]

    return np.array([R_findex_ankle, R_ankle_knee, R_knee_hip, R_hip_shoulder, R_index_wrist, R_wrist_elbow, R_elbow_shoulder,
                     L_findex_ankle, L_ankle_knee, L_knee_hip, L_hip_shoulder, L_index_wrist, L_wrist_elbow, L_elbow_shoulder,
                     lshoulder_rshoulder, lhip_rhip]).transpose(1, 0, 2)

def cosine_similarity(vec1: np.array, vec2: np.array, weight: np.array, min_cosine_sim: int, expo: bool):
    """
    คำนวณความเหมือนด้วย cosine similarity (min_cosine_sim - 1)
    K(X, Y) = X dot Y / (||X||*||Y||)

    input:
        vec1: numpy array ที่ข้างในมีเวกเตอร์ท่อนแขนท่อนขา 16 ท่อนของวิดิโอที่ 1 ได้จากฟังก์ชัน find_limb
        vec2: numpy array ที่ข้างในมีเวกเตอร์ท่อนแขนท่อนขา 16 ท่อนของวิดิโอที่ 2 ได้จากฟังก์ชัน find_limb
        weight : ค่าถ่วงน้ำหนัก ที่ได้จากฟังก์ชัน gen_weight
        min_cosine_sim: ค่าที่ใช้การ normalize แต่ละท่อนแขนท่อนขา รับเข้ามาเป็นองศา

    output:
         numpy array ความเหมือนของเวกเตอร์ 16 ท่อนที่คูณกับค่าถ่วงน้ำหนักแล้ว มีช่วงตั้งแต่ 0 - 1
    """
    # 180 คือ -1, 90 คือ 0, 45 คือ sqrt(2)/2
    min_cosine_sim = np.cos(np.radians(min_cosine_sim))

    limb_diff_norm = (((np.sum(vec1*vec2, axis=1)/(np.linalg.norm(vec1, axis=1)
                      * np.linalg.norm(vec2, axis=1)))-min_cosine_sim)/(1-min_cosine_sim))

    if not expo:
        return limb_diff_norm * weight
    else:
        limb_diff_expo = np.where((limb_diff_norm >= 0) & (
            limb_diff_norm <= 1), 1/(1+(limb_diff_norm/(1-limb_diff_norm))**-2), limb_diff_norm)
        # each limb has range between 0 and 1 (default weight)
        return limb_diff_expo * weight


def cosine_difference(vec1: np.array, vec2: np.array, weight: np.array, max_cosine_diff: int, expo: bool):
    """
    คำนวณความต่างด้วย 1 - cosine similarity (0 - max_cosine_diff)

    input:
        vec1: numpy array ที่ข้างในมีเวกเตอร์ท่อนแขนท่อนขา 16 ท่อนของวิดิโอที่ 1 ได้จากฟังก์ชัน find_limb
        vec2: numpy array ที่ข้างในมีเวกเตอร์ท่อนแขนท่อนขา 16 ท่อนของวิดิโอที่ 2 ได้จากฟังก์ชัน find_limb
        max_cosine_diff: ค่าที่ใช้การ normalize แต่ละท่อนแขนท่อนขา
        weight : ค่าถ่วงน้ำหนัก ที่ได้จากฟังก์ชัน gen_weight

    output:
        numpy array ความต่างของเวกเตอร์ 16 ท่อนที่คูณกับค่าถ่วงน้ำหนักแล้ว มีช่วงตั้งแต่ 0 - 1
    """

    # 180 คือ -1, 90 คือ 0, 45 คือ sqrt(2)/2
    max_cosine_diff = np.cos(np.radians(max_cosine_diff))

    limb_diff_norm = (1 - (np.sum(vec1*vec2, axis=1)/(np.linalg.norm(vec1,
                      axis=1)*np.linalg.norm(vec2, axis=1))))/(1-max_cosine_diff)

    if not expo:
        return limb_diff_norm * weight
    else:
        limb_diff_expo = np.where((limb_diff_norm >= 0) & (
            limb_diff_norm <= 1), 1/(1+(limb_diff_norm/(1-limb_diff_norm))**-2), limb_diff_norm)
        # each limb has range between 0 and 1 (default weight)
        return limb_diff_expo * weight