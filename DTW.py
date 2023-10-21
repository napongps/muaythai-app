import numpy as np
from angle import angle_difference, angle_similarity
from cosine import cosine_difference, cosine_similarity
from MAW import *

def dm(extracted_ladk_vid1: np.array, extracted_ladk_vid2: np.array, sim_diff_function,
       weight, MAW, norm_value: int, windows: int, expo=False):

    N = extracted_ladk_vid1.shape[0]
    M = extracted_ladk_vid2.shape[0]

    dist_mat = np.zeros((N, M))
    dist_ladk_mat = np.zeros((N, M), dtype=object)

    if MAW:
        weight = MA_W(extracted_ladk_vid1, windows, norm_value)
    else:
        weight = [weight]*max(N, M)

    for i in range(N):
        for j in range(M):

            dist_ladk_mat[i, j] = sim_diff_function(
                extracted_ladk_vid1[i], extracted_ladk_vid2[j], weight[i], norm_value, expo)
            dist_mat[i, j] = np.sum(dist_ladk_mat[i, j]) / \
                np.sum(weight[i])  # 0-1 (Default weight)

    return dist_ladk_mat, dist_mat

def dp(dist_mat: np.array, mode: str):
    """
    ขวาล่าง-ซ้ายบน (cost)
    Find minimum-cost path through matrix `dist_mat` using dynamic programming.

    The cost of a path is defined as the sum of the matrix entries on that
    path. See the following for details of the algorithm:

    - http://en.wikipedia.org/wiki/Dynamic_time_warping
    - https://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/dp.m

    The notation in the first reference was followed, while Dan Ellis's code
    (second reference) was used to check for correctness. Returns a list of
    path indices and the cost matrix.
    """

    N, M = dist_mat.shape

    # Initialize the cost matrix
    cost_mat = np.zeros((N, M))
    if 'difference' in mode:
        cost_mat = np.insert(cost_mat, 0, np.inf, axis=1)
        cost_mat = np.insert(cost_mat, 0, np.inf, axis=0)
    elif 'similarity' in mode:
        cost_mat = np.insert(cost_mat, 0, -np.inf, axis=1)
        cost_mat = np.insert(cost_mat, 0, -np.inf, axis=0)
    cost_mat[0, 0] = 0

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M), dtype='uint8')
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            if 'difference' in mode:
                i_penalty = np.argmin(penalty)
            elif 'similarity' in mode:
                i_penalty = np.argmax(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]
    vert_hor = 0

    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]

        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1

        elif tb_type == 1:
            # Insertion
            i = i - 1
            vert_hor += 1

        elif tb_type == 2:
            # Deletion
            j = j - 1
            vert_hor += 1

        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    cost_mat = cost_mat/len(path)

    return path[::-1], cost_mat, vert_hor

def unique_path(path):
    new_path = []
    temp_i = None
    temp_j = None
    for i, j in path:
        if (i == temp_i) or (j == temp_j):
            continue
        temp_i = i
        temp_j = j
        new_path.append((i, j))
    return np.array(new_path)

def dtw(extracted_ladk_vid1: np.array, extracted_ladk_vid2: np.array, sim_diff_function,
        weight, MAW, norm_value=180, windows=50, thresh=False, expo=False):

    dist_lndmk_mat, dist_mat = dm(extracted_ladk_vid1=extracted_ladk_vid1,
                                  extracted_ladk_vid2=extracted_ladk_vid2,
                                  sim_diff_function=sim_diff_function,
                                  weight=weight,
                                  MAW=MAW,
                                  norm_value=norm_value,
                                  windows=windows,
                                  expo=expo)

    path, cost_mat, vert_hor = dp(dist_mat=dist_mat,
                                           mode=sim_diff_function.__name__)

    row = np.array(path)[:, 0]
    col = np.array(path)[:, 1]
    dist_mat_path = dist_mat[row, col]

    unique_row = unique_path(path)[:, 0]
    unique_col = unique_path(path)[:, 1]
    dist_mat_unique = dist_mat[unique_row, unique_col]

    frames_count = dist_mat_unique.shape[0]

    if 'difference' in sim_diff_function.__name__:
        dist_mat_path = 1-dist_mat_path
        dist_mat_unique = 1-dist_mat_unique

    cost = np.sum(dist_mat_path)/len(path)
    std_all = np.std(dist_mat_path, dtype=np.float32)
    std_unique = np.std(dist_mat_unique, dtype=np.float32)
    mean = np.mean(dist_mat_path, dtype=np.float32)

    if thresh:
        thresh = np.mean(dist_mat_path)-0.03
        cost = np.sum(dist_mat_path[dist_mat_path > thresh])/len(path)

    return path, dist_mat, dist_lndmk_mat, cost_mat, cost

