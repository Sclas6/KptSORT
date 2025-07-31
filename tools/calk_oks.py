import torch
import math
import numpy as np
from numba import jit

def kpt2area(kpt_gt: list):
    list_x = [xy[0] for xy in kpt_gt if not math.isnan(xy[0])]
    list_y = [xy[1] for xy in kpt_gt if not math.isnan(xy[1])]
    size = (max(list_x) - min(list_x)) * (max(list_y) - min(list_y))
    return size

def center(kpt_gt: list):
    list_x = [xy[0] for xy in kpt_gt if xy is not None]
    list_y = [xy[1] for xy in kpt_gt if xy is not None]
    return (int((min(list_x) + max(list_x)) / 2) + 20, int((min(list_y) + max(list_y)) / 2))

@jit(nopython=True, cache=True)
def bin2(n: int) -> list:
    result = []
    while True:
        result.append(int(n % 2))
        n /= 2
        if n <= 1: break
    while True:
        if len(result) == 6:
            break
        result.append(0)
    return result[::-1]

@jit(nopython=True, cache=True)
def oks(gt_kpts: list, pred_kpts: list, sigma) -> float:
    gt_kpts_m = []
    pred_kpts_m = []
    mask_gt = bin2(gt_kpts[6])
    #mask_pred = str(bin(int(pred_kpts[6])))[2:].zfill(6)
    for i in range(0, len(gt_kpts) - 1, 2):
        if mask_gt[i] == 0:
            gt_kpts_m.append([gt_kpts[i], gt_kpts[i + 1]])
        else:
            gt_kpts_m.append([np.nan, np.nan])
    for i in range(0, len(gt_kpts) - 1, 2):
        pred_kpts_m.append([pred_kpts[i], pred_kpts[i + 1]])
    oks = 0
    epsilon = np.finfo(np.float32).eps

    top = 0
    bottom = 0
    #area = kpt2area(gt_kpts_m) * 0.053
    area = 150
    #print(area)
    for i, gt_kpt in enumerate(gt_kpts_m):
        #print(gt_kpt)
        if np.isnan(gt_kpt[0]) or np.isnan(pred_kpts_m[i][0]):
            continue
        dist_sq = (gt_kpt[0] - pred_kpts_m[i][0])**2 + (gt_kpt[1] - pred_kpts_m[i][1])**2
        top += math.exp(-dist_sq / (2 * area**2 * sigma**2 + epsilon))
        bottom += 1
    oks = top / (bottom + epsilon)
    return oks