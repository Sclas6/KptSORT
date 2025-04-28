import numpy as np
import csv
import copy
import random
from itertools import permutations
from tools.calk_oks import oks
import cv2

BODYPARTS = 3
INDIVISUALS = 20

def check_overlap(kpts:list, indivisuals:list, threshould, raw: bool=True):
    if raw:
        kpts_new = []
        for kpt in kpts:
            for i, c in enumerate(kpt):
                if i == 2: continue
                kpts_new.append(c)
        kpts_new.append(0)
        kpts = np.array(kpts_new)
        indivisuals_new = []
        for individual in indivisuals:
            tmp = []
            for i_kpts in individual:
                for i, c in enumerate(i_kpts):
                    if i == 2: continue
                    tmp.append(c)
            tmp.append(0)
            indivisuals_new.append(tmp)
        indivisuals = np.array(indivisuals_new)
        
    oks_max = 0
    for individual in indivisuals:
        score = oks(kpts, individual, 0.1)
        oks_max = max(oks_max, score)
    if oks_max > threshould:
        return False
    return True


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

def take_difference(parts_raw: list, data_csv, frame: int):
    """
    parts_raw: [{x, y, p}: head, {x, y, p}: body, {x, y, p}: sting]
    """
    # indivisual have full body kpts
    parts_full = [set(), set(), set()]
    # full kpts
    parts_diff = copy.deepcopy(parts_raw)
    indivisuals = np.zeros((len(data_csv[frame]), 7))
    for i, part in enumerate(data_csv[frame]):
        #indivisuals.append([[part[0][0], part[0][1]], [part[1][0], part[1][1]], [part[2][0], part[2][1]]])
        #print(np.array([[part[0][0], part[0][1]], [part[1][0], part[1][1]], [part[2][0], part[2][1]]]))
        indivisuals[i] = np.array([part[0][0], part[0][1], part[1][0], part[1][1], part[2][0], part[2][1], 0])
        #print(part)
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

def assemble_w_yolo(model, frame, data_pkl, data_csv, threshould, id_frame):
    parts_raw = pkl2setlist(data_pkl, id_frame)
    parts_diff, individuals = take_difference(parts_raw, data_csv, id_frame)
    results = model.predict(frame, device = 0, conf = 0.1, verbose=False)
    bboxes = [result.boxes.xyxy[0].tolist() for result in results[0]]

    for i, individual in enumerate(data_csv[id_frame]):
        """
        for kpt in individual:
            for bbox in bboxes:
                if kpt_in_bbox(kpt, bbox, 0):
                    bboxes.remove(bbox)
                    break
        """
        for bbox in bboxes:
            for individusl in data_csv[id_frame]:
                rm = 0
                for kpt in individual:
                    if kpt_in_bbox(kpt, bbox, 0):
                        rm += 1
                if rm == 3:
                    bboxes.remove(bbox)
                    break
        #"""

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
            if check_overlap(individual, individuals, threshould, False):
                individuals = np.concatenate([individuals, [individual]])
                pass
    #print(individuals)
    
    img = results[0].plot(conf=False, labels=False)
    boxes = results[0].boxes
    h, w, _ = frame.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].xyxy.tolist()[0]
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)

    img[mask == 0] = 0
    
    return individuals, img