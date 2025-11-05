import numpy as np
import csv
import copy
import random
from itertools import permutations
from tools.calk_oks import oks
from numba import njit
from numba import jit
import numba
import math

@njit
def delete_3d_row(arr, num):
    result = np.empty((len(arr)-1, 4, 2), arr.dtype)
    c = 0
    for i, a in enumerate(arr):
        if i != num:
            result[c] = a
            c += 1
    return result

@njit
def delete_2d_row(arr, num):
    result = np.empty((len(arr)-1, 3), arr.dtype)
    c = 0
    for i, a in enumerate(arr):
        if i != num:
            result[c] = a
            c += 1
    return result

@njit
def concatnate_2d(arr1, arr2):
    result = np.empty((len(arr1) + len(arr2), len(arr1[0])), arr1.dtype)
    c = 0
    for a in arr1:
        result[c] = a
        c += 1
    for a in arr2:
        result[c] = a
        c += 1
    return result

@njit
def check_overlap(kpts:list, indivisuals:list, threshould):
    oks_max = 0
    for individual in indivisuals:
        score = oks(kpts, individual, 0.1)
        oks_max = max(oks_max, score)
    if oks_max > threshould:
        return False
    return True


def load_csv(path, n_indivisuals, n_bodyparts):
    csv_data = np.genfromtxt(path, delimiter=',', dtype=np.float32, skip_header=4, usecols=[i for i in range(1, n_indivisuals*9+1)])
    indivisuals = []
    for row in csv_data:
        kpts = np.reshape(row, (int(len(row)/9), 3, 3))
        kpts = kpts[~np.all(np.isnan(kpts), axis=(1, 2))]
        indivisuals.append(kpts)
    return indivisuals


def pkl2setlist(pkl: dict, frame: int) -> set:
    def _format_frame_string(digits, id_frame):
        return "".join(["0" for _ in range(digits - len(str(id_frame)))]) + str(id_frame)
    tmp = []
    digits = len(list(pkl.keys())[1][5:])
    for part in zip(pkl[f"frame{_format_frame_string(digits, frame)}"]["coordinates"][0], np.squeeze(pkl[f"frame{_format_frame_string(digits, frame)}"]["confidence"])):
        mask = part[1] < 0.6
        tmp.append(list(zip(part[0][~mask].tolist(), part[1][~mask].tolist())))
    parts = []
    part_num = 0
    for part in tmp:
        part_set = []
        for kpt in part:
            #np.append(parts, (kpt[0][0], kpt[0][1], kpt[1][0]))
            part_set.append((kpt[0][0], kpt[0][1], kpt[1]))
        parts.append(part_set)
        part_num = max(part_num, len(part_set))
    for part in parts:
        while len(part) != part_num:
            part.append((np.nan, np.nan, np.nan))
    return np.array(parts)

@njit(cache=True)
def np_all_axis1(x):
    """Numba compatible version of np.all(x, axis=1)."""
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_and(out, x[:, i])
    return out

## TODO
def take_difference(parts_raw, data_csv):
    """
    parts_raw: [{x, y, p}: head, {x, y, p}: body, {x, y, p}: sting]
    """
    # indivisual have full body kpts
    parts_full = [set(), set(), set()]
    # full kpts
    parts_diff = parts_raw.copy()
    #indivisuals = np.zeros((len(data_csv), 7))
    indivisuals = np.empty((0, 7))
    dist_avg = 0
    for i, part in enumerate(data_csv):
        dist_avg += np.linalg.norm(np.array([part[0][0], part[0][1]]) - np.array([part[1][0], part[1][1]])) + np.linalg.norm(np.array([(part[1][0], part[1][1])]) - np.array([(part[2][0], part[2][1])]))
        #indivisuals.append([[part[0][0], part[0][1]], [part[1][0], part[1][1]], [part[2][0], part[2][1]]])
    dist_avg /= i
    for i, part in enumerate(data_csv):
        if not np.linalg.norm(np.array([part[0][0], part[0][1]]) - np.array([part[1][0], part[1][1]])) + np.linalg.norm(np.array([(part[1][0], part[1][1])]) - np.array([(part[2][0], part[2][1])])) > dist_avg * 1.5:
            indivisuals = np.append(indivisuals, np.array([[part[0][0], part[0][1], part[1][0], part[1][1], part[2][0], part[2][1], 0]]), axis=0)
            for j, kpt in enumerate(part):
                parts_full[j].add(tuple(kpt.tolist()))
        else:
            for j, kpt in enumerate(part):
                parts_diff[j].add(tuple(kpt.tolist()))

    for i in range(len(parts_diff)):
        parts_diff[i] -= parts_full[i]
    for i, part in enumerate(copy.deepcopy(parts_diff)):
        for kpt in part:
            if kpt[2] < 0.4:
                parts_diff[i].remove(kpt)
    return parts_diff, indivisuals

@njit
def take_difference_jit(parts_raw, data_csv):
    parts_csv = np.full((3, len(data_csv), 3), np.nan)
    parts_dif = np.copy(parts_raw)
    individuals = np.empty((0, 7))
    
    dist_avg = 0
    for i, part in enumerate(data_csv):
        part_float = part.astype(np.float32)
        dist_avg += (np.linalg.norm(part_float[0] - part_float[1]) + np.linalg.norm(part_float[1] - part_float[2]))
    dist_avg /= len(data_csv)
    for i, part in enumerate(data_csv):
        part_float = part.astype(np.float32)
        
        #if not (np.linalg.norm(part_float[0] - part_float[1]) + np.linalg.norm(part_float[1] - part_float[2])) > 55:
        if not (np.linalg.norm(part_float[0] - part_float[1]) + np.linalg.norm(part_float[1] - part_float[2])) > dist_avg * 1.5:
            individuals = np.append(individuals, np.array([[part[0][0], part[0][1], part[1][0], part[1][1], part[2][0], part[2][1], 0]]), axis=0)
            for j, kpt in enumerate(part):
                parts_csv[j,i] = kpt
        else:
            parts_dif_new = np.full((parts_dif.shape[0], parts_dif.shape[1] + 1, parts_dif.shape[2]), np.nan)
            parts_dif_new[:,:parts_dif.shape[1],:] = parts_dif
            for j, kpt in enumerate(part):
                parts_dif_new[j, -1] = kpt
            parts_dif = parts_dif_new
                


    for i, part in enumerate(parts_raw):
        for j, kpt in enumerate(part):
            if np.any(np_all_axis1(kpt == parts_csv[i])):
                parts_dif[i, j] = np.array([np.nan, np.nan, np.nan])

    return parts_dif, individuals

@jit(nopython=True, cache=True)
def kpt_in_bbox(kpt, xyxy, margin):
    if (kpt[0] > min(xyxy[0], xyxy[2]) - margin and kpt[0] < max(xyxy[0], xyxy[2]) + margin) and (kpt[1] > min(xyxy[1], xyxy[3]) - margin and kpt[1] < max(xyxy[1], xyxy[3]) + margin):
        return True
    else:
        return False
    
def gen_random_colors(length: int, seed: int):
    random.seed(seed)
    colors = set()
    while True:
        colors.add((random.randint(50, 255), random.randint(50, 255), random.randint(150, 255)))
        if len(colors) == length: return list(colors)

@njit
def shortning_rect(fixed: np.ndarray, other: np.ndarray, factor: float):
    fixed_p1 = fixed[0]
    fixed_p2 = fixed[1]
    other_p1 = other[0]
    other_p2 = other[1]

    fixed_vector = (fixed_p2[0] - fixed_p1[0], fixed_p2[1] - fixed_p1[1])
    vertical_vector = (-fixed_vector[1], fixed_vector[0])
    vertical_vector_magnitude = math.sqrt(vertical_vector[0]**2 + vertical_vector[1]**2)    
    unit_vertical_vector = (vertical_vector[0] / vertical_vector_magnitude,
                            vertical_vector[1] / vertical_vector_magnitude) 
    shortening_factor = factor
    new_p1 = fixed_p1
    new_p2 = fixed_p2   
    movement = np.linalg.norm(fixed[0]-other[0]) * -(1 - shortening_factor)   
    new_other_p1 = np.array([other_p1[0] - unit_vertical_vector[0] * movement,
                    other_p1[1] - unit_vertical_vector[1] * movement])
    new_other_p2 = np.array([other_p2[0] - unit_vertical_vector[0] * movement,
                    other_p2[1] - unit_vertical_vector[1] * movement])
    result = np.empty((4, 2))
    result[0], result[1], result[2], result[3] = new_p1, new_p2, new_other_p1, new_other_p2
    return result

@njit
def get_enlarged_rectangle(vertices, scale_x, scale_y):
    center = np.sum(vertices, axis=0) / len(vertices)
    edge1 = vertices[1] - vertices[0]
    edge2 = vertices[3] - vertices[0]
    length1 = np.linalg.norm(edge1)
    length2 = np.linalg.norm(edge2)
    relative_coords = vertices - center
    new_vertices = np.empty((4, 2))
    for i, relative_coord in enumerate(relative_coords):
        relative_coord = np.copy(relative_coord)
        local_x = np.dot(relative_coord, edge1 / length1) if length1 > 1e-6 else 0
        local_y = np.dot(relative_coord, edge2 / length2) if length2 > 1e-6 else 0
        scaled_local_x = local_x * scale_x
        scaled_local_y = local_y * scale_y
        new_global_coord = center + (edge1 / length1) * scaled_local_x + (edge2 / length2) * scaled_local_y
        new_vertices[i] = new_global_coord
    return new_vertices

@njit
def kpt_in_rect(point, rect_vertices):
    n = len(rect_vertices)
    signs = []
    for i in range(n):
        p1 = rect_vertices[i]
        p2 = rect_vertices[(i + 1) % n]
        edge = p2 - p1
        vec_to_point = point - p1
        cross_product = numba.np.arraymath._cross2d_operation(edge, vec_to_point)
        signs.append(np.sign(cross_product))

    first_sign = 0
    for sign in signs:
        if sign != 0:
            first_sign = sign
            break

    return np.all(np.array([sign == 0 or sign == first_sign for sign in signs]))

@njit
def assemble_w_yolo(xyxys, data_pkl, data_csv, threshould):
    def _array2bin(arr: np.ndarray) -> int:
        n = len(arr) - 1
        result = 0
        for a in arr:
            result += a * 2**n
            n -= 1
        return result

    parts_diff, individuals = take_difference_jit(data_pkl, data_csv)
    for individual in data_csv:
        for i, xyxy in enumerate(np.copy(xyxys)):
            top = np.array([[xyxy[0][0], xyxy[0][1]], [xyxy[1][0], xyxy[1][1]]])
            bottom = np.array([[xyxy[2][0], xyxy[2][1]], [xyxy[3][0], xyxy[3][1]]])
            rect_top = get_enlarged_rectangle(shortning_rect(top, bottom, 0.2), 1, 1)
            rect_bottom = get_enlarged_rectangle(shortning_rect(bottom, top, 0.2), 1, 1)
            rm = 0
            
            """
            cv2.polylines(frame, [rect_top.astype(np.int32)], True, (0, 255, 255), 4)
            cv2.circle(frame, (int(individual[0][0]), int(individual[0][1])), 10, (0, 255, 255), 5)
            cv2.polylines(frame, [rect_bottom.astype(np.int32)], True, (255, 255, 0), 4)
            cv2.circle(frame, (int(individual[2][0]), int(individual[2][1])), 10, (255, 255, 0), 5)
            """
            
            if kpt_in_rect(np.array([individual[0][0], individual[0][1]]), rect_top):
                rm += 1
            if kpt_in_rect(np.array([individual[2][0], individual[2][1]]), rect_bottom):
                rm += 1
            if rm == 2:
                xyxys = delete_3d_row(xyxys, i)
                break

    for xyxy in xyxys:
        #individual = np.full((3, 2), 0)
        individual = np.full((7, ), np.nan)
        for i, part in enumerate(np.copy(parts_diff)):
            for j, kpt in enumerate(part):
                if np.any(np.isnan(kpt)): continue
                #if kpt_in_rect((kpt[0], kpt[1]), get_enlarged_rectangle(np.array(xyxy), 1.5, 1.2)):
                if kpt_in_rect(np.array([kpt[0], kpt[1]]), get_enlarged_rectangle(xyxy, 1, 1)):
                    individual[i * 2] = kpt[0]
                    individual[i * 2 + 1] = kpt[1]
                    #parts_diff[i].remove(kpt)
                    parts_diff[i][j] = np.array([np.nan, np.nan, np.nan])
        mask_list = np.where(np.isnan(individual[:6]), 1, 0)
        #individual[6] = int("".join([f"{c}" for c in mask_list.tolist()]), 2)
        individual[6] = _array2bin(mask_list)
        if not np.all(np.isnan(individual[:6])):
            if check_overlap(individual, individuals, threshould):
                #print(individual)
                individuals = concatnate_2d(individuals, [individual])

    return individuals