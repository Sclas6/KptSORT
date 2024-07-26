import numpy as np
import csv
import copy
import random

BODYPARTS = 3
INDIVISUALS = 28

def load_csv(path):
    with open(path) as f:
        reader = csv.reader(f)
        indivisual = []
        for i, row in enumerate(reader):
            if i <=3: continue
            tmp = []
            for j in range(INDIVISUALS):
                kpts = []
                for k in range((BODYPARTS+6) * j + 1, (BODYPARTS+6) * (j + 1) + 1, 3):
                    if row[k] == "":
                        pass
                    else:
                        kpts.append((float(row[k]), float(row[k + 1]), float(row[k + 2])))
                    k += 1
                if len(kpts) != 0:
                    tmp.append(kpts)
            indivisual.append(tmp)
    return indivisual

def pkl2setlist(pkl: dict, frame: int) -> set:
    tmp = []
    for part in zip(pkl[f"frame{frame:04}"]["coordinates"][0], pkl[f"frame{frame:04}"]["confidence"]):
        tmp.append(list(zip(part[0].tolist(), part[1].tolist())))
    parts = []
    for part in tmp:
        part_set = set()
        for kpt in part:
            part_set.add((kpt[0][0], kpt[0][1], kpt[1][0]))
        parts.append(part_set)
    return parts

def take_difference(parts_raw: list, data_csv, frame: int):
    parts_full = [set(), set(), set()]
    parts_diff = copy.deepcopy(parts_raw)
    indivisuals = np.zeros((len(data_csv[frame]), 7))
    for i, part in enumerate(data_csv[frame]):
        #indivisuals.append([[part[0][0], part[0][1]], [part[1][0], part[1][1]], [part[2][0], part[2][1]]])
        #print(np.array([[part[0][0], part[0][1]], [part[1][0], part[1][1]], [part[2][0], part[2][1]]]))
        indivisuals[i] = np.array([part[0][0], part[0][1], part[1][0], part[1][1], part[2][0], part[2][1], 0])
        for j, kpt in enumerate(part):
            parts_full[j].add(kpt)
    for i in range(len(parts_diff)):
        parts_diff[i] -= parts_full[i]
    for i, part in enumerate(copy.deepcopy(parts_diff)):
        for kpt in part:
            if kpt[2] < 0.8:
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

def assemble_w_yolo(model, frame, data_pkl, data_csv, id_frame):
    parts_raw = pkl2setlist(data_pkl, id_frame)
    parts_diff, individuals = take_difference(parts_raw, data_csv, id_frame)
    results = model.predict(frame, device = 0, conf = 0.1, verbose=False)
    bboxes = [result.boxes.xyxy[0].tolist() for result in results[0]]
    for individual in data_csv[id_frame]:
        for kpt in individual:
            for bbox in bboxes:
                if kpt_in_bbox(kpt, bbox, 5):
                    bboxes.remove(bbox)
                    break
    for bbox in bboxes:
        #individual = np.full((3, 2), 0)
        individual = np.full((7, ), np.nan)
        for i, part in enumerate(parts_diff):
            for kpt in part:
                if kpt_in_bbox(kpt, bbox, 0):
                    individual[i * 2] = kpt[0]
                    individual[i * 2 + 1] = kpt[1]
        mask_list = np.where(np.isnan(individual[:6]), 1, 0)
        individual[6] = int("".join([f"{c}" for c in mask_list.tolist()]), 2)
        if not np.all(np.isnan(individual[:6])): 
            individuals = np.concatenate([individuals, [individual]])
    #print(individuals)
    return individuals