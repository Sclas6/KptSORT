import numpy as np
import csv
import copy
import random
from itertools import permutations
from tools.calk_oks import oks
import math

def check_overlap(kpts:list, indivisuals:list, threshould):
    oks_max = 0
    for individual in indivisuals:
        score = oks(kpts, individual, 0.1)
        oks_max = max(oks_max, score)
    if oks_max > threshould:
        return False
    return True


def load_csv(path, n_indivisualt, n_bodyparts):
    with open(path) as f:
        reader = csv.reader(f)
        indivisual = []
        for i, row in enumerate(reader):
            if i <=3: continue
            tmp = []
            for j in range(n_indivisualt):
                kpts = []
                for k in range((n_bodyparts+6) * j + 1, (n_bodyparts+6) * (j + 1) + 1, 3):
                    if row[k] == "":
                        pass
                    else:
                        kpts.append((float(row[k]), float(row[k + 1]), float(row[k + 2])))
                    k += 1
                if len(kpts) != 0:
                    #if not check_overlap(kpts, tmp, True):
                        tmp.append(kpts)
            indivisual.append(tmp)
    return indivisual


def pkl2setlist(pkl: dict, frame: int) -> set:
    def _format_frame_string(digits, id_frame):
        return "".join(["0" for _ in range(digits - len(str(id_frame)))]) + str(id_frame)
    tmp = []
    digits = len(list(pkl.keys())[1][5:])
    for part in zip(pkl[f"frame{_format_frame_string(digits, frame)}"]["coordinates"][0], pkl[f"frame{_format_frame_string(digits, frame)}"]["confidence"]):
        tmp.append(list(zip(part[0].tolist(), part[1].tolist())))
    parts = []
    for part in tmp:
        part_set = set()
        for kpt in part:
            part_set.add((kpt[0][0], kpt[0][1], kpt[1][0]))
        parts.append(part_set)
    return parts

#@jit(nopython=True, cache=True)
def take_difference(parts_raw: list, data_csv):
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
                parts_full[j].add(kpt)
        else:
            for j, kpt in enumerate(part):
                parts_diff[j].add(kpt)

    for i in range(len(parts_diff)):
        parts_diff[i] -= parts_full[i]
    for i, part in enumerate(copy.deepcopy(parts_diff)):
        for kpt in part:
            if kpt[2] < 0.4:
                parts_diff[i].remove(kpt)
    return parts_diff, indivisuals

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
    return [new_p1, new_p2, new_other_p1, new_other_p2]

def get_enlarged_rectangle(vertices, scale_x, scale_y):
    center = np.sum(vertices, axis=0) / len(vertices)
    edge1 = vertices[1] - vertices[0]
    edge2 = vertices[3] - vertices[0]
    length1 = np.linalg.norm(edge1)
    length2 = np.linalg.norm(edge2)
    relative_coords = vertices - center
    new_vertices = []
    for relative_coord in relative_coords:
        local_x = np.dot(relative_coord, edge1 / length1) if length1 > 1e-6 else 0
        local_y = np.dot(relative_coord, edge2 / length2) if length2 > 1e-6 else 0
        scaled_local_x = local_x * scale_x
        scaled_local_y = local_y * scale_y
        new_global_coord = center + (edge1 / length1) * scaled_local_x + (edge2 / length2) * scaled_local_y
        new_vertices.append(new_global_coord)

    return np.array(new_vertices)

def kpt_in_rect(point, rect_vertices):
    n = len(rect_vertices)
    signs = []
    for i in range(n):
        p1 = rect_vertices[i]
        p2 = rect_vertices[(i + 1) % n]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        vec_to_point = point - p1
        cross_product = np.cross(edge, vec_to_point) # 2Dでの外積はスカラー値
        signs.append(np.sign(cross_product))

    first_sign = 0
    for sign in signs:
        if sign != 0:
            first_sign = sign
            break
        
    return all(sign == 0 or sign == first_sign for sign in signs)

def assemble_w_yolo(model, frame, data_pkl, data_csv, threshould):
    parts_diff, individuals = take_difference(data_pkl, data_csv)
    results = model.predict(frame, device=0, conf=0.45, verbose=False)
    if vars(model)["task"] == "detect":
        bboxes = [result.boxes.xyxy[0].tolist() for result in results[0]]

        for i, individual in enumerate(data_csv):
            for bbox in bboxes:
                for individusl in data_csv:
                    rm = 0
                    for kpt in individual:
                        if kpt_in_bbox(kpt, bbox, 0):
                            rm += 1
                    if rm == 3:
                        bboxes.remove(bbox)
                        break

        for bbox in bboxes:
            #individual = np.full((3, 2), 0)
            individual = np.full((7, ), np.nan)
            for i, part in enumerate(copy.deepcopy(parts_diff)):
                for kpt in part:
                    if kpt_in_bbox(kpt, bbox, 5):
                        individual[i * 2] = kpt[0]
                        individual[i * 2 + 1] = kpt[1]
                        parts_diff[i].remove(kpt)
            mask_list = np.where(np.isnan(individual[:6]), 1, 0)
            individual[6] = int("".join([f"{c}" for c in mask_list.tolist()]), 2)
            if not np.all(np.isnan(individual[:6])):
                if check_overlap(individual, individuals, threshould):
                    individuals = np.concatenate([individuals, [individual]])
    #print(individuals)
    elif vars(model)["task"] == "obb":
        xyxys = [result.obb.xyxyxyxy.tolist() for result in results][0]
        data_csv = set((tuple(individual) for individual in data_csv))
        for individual in data_csv:
            for xyxy in xyxys:
                #for individusl in data_csv:
                    top = np.array([[xyxy[0][0], xyxy[0][1]], [xyxy[1][0], xyxy[1][1]]])
                    bottom = np.array([[xyxy[2][0], xyxy[2][1]], [xyxy[3][0], xyxy[3][1]]])
                    #rect_top = get_enlarged_rectangle(shortning_rect(top, bottom, 0.2), 1.2, 2.5)
                    #rect_bottom = get_enlarged_rectangle(shortning_rect(bottom, top, 0.2), 1.2, 2.5)
                    rect_top = get_enlarged_rectangle(shortning_rect(top, bottom, 0.2), 1, 1)
                    rect_bottom = get_enlarged_rectangle(shortning_rect(bottom, top, 0.2), 1, 1)
                    rm = 0
                    
                    """
                    cv2.polylines(frame, [rect_top.astype(np.int32)], True, (0, 255, 255), 4)
                    cv2.circle(frame, (int(individual[0][0]), int(individual[0][1])), 10, (0, 255, 255), 5)
                    cv2.polylines(frame, [rect_bottom.astype(np.int32)], True, (255, 255, 0), 4)
                    cv2.circle(frame, (int(individual[2][0]), int(individual[2][1])), 10, (255, 255, 0), 5)
                    """
                    
                    if kpt_in_rect((individual[0][0], individual[0][1]), rect_top):
                        rm += 1
                    if kpt_in_rect((individual[2][0], individual[2][1]), rect_bottom):
                        rm += 1
                    if rm == 2:
                        xyxys.remove(xyxy)
                        break

        for xyxy in xyxys:
            #individual = np.full((3, 2), 0)
            individual = np.full((7, ), np.nan)
            for i, part in enumerate(copy.deepcopy(parts_diff)):
                for kpt in part:
                    #if kpt_in_rect((kpt[0], kpt[1]), get_enlarged_rectangle(np.array(xyxy), 1.5, 1.2)):
                    if kpt_in_rect((kpt[0], kpt[1]), get_enlarged_rectangle(np.array(xyxy), 1, 1)):
                        individual[i * 2] = kpt[0]
                        individual[i * 2 + 1] = kpt[1]
                        parts_diff[i].remove(kpt)
            mask_list = np.where(np.isnan(individual[:6]), 1, 0)
            individual[6] = int("".join([f"{c}" for c in mask_list.tolist()]), 2)
            if not np.all(np.isnan(individual[:6])):
                if check_overlap(individual, individuals, threshould):
                    individuals = np.concatenate([individuals, [individual]])

    img = results[0].plot()
    
    return individuals, img