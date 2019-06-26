import numpy as np


def compute_iou_matrix(d, g):
    len_d = len(d)
    len_g = len(g)
    ret = np.zeros((len_d, len_g))
    for i in range(len_d):
        for j in range(len_g):
            ret[i][j] = compute_iou(d[i], g[j])
    return ret


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (x1, y1, x2, y2)
    :param rec2: (x1, y1, x2, y2)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)
