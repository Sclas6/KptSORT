import os
os.chdir("/kpsort")
import cv2
from tools.kpsort import Sort
from tools.loadpkl import *
from tools.calk_oks import oks
from tools.AssignBeeHive import AssignBeeHive
from ultralytics import YOLO
import numpy as np
import pickle
import math
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(suppress=True)

def check_overlap_2(individuals, threshold: int):
    fulls_sorted = list()
    fulls = dict()
    for i, individual in enumerate(individuals):
        if np.any(np.isnan(individual)):
            pass
        else:
            tmp = list()
            for j in range(0, len(individual) - 1, 2):
                tmp.append((individual[j], individual[j + 1]))
            tmp = sorted(tmp)
            fulls_sorted.append(np.append(np.array(tmp).flatten(), 0))
            fulls[tuple(np.append(np.array(tmp).flatten(), 0).tolist())] = i
    desirable2remove = list()
    for i, individual in enumerate(fulls_sorted):
        for j, ind in enumerate(fulls_sorted):
            if i >= j: continue
            oks_value = oks(individual, ind, 0.1)
            if oks_value > threshold:
                desirable2remove.append((fulls[tuple(individual)], fulls[tuple(ind)]))
    return np.array(list(desirable2remove))

def mark_losted_trackers(frame, trackers, ids_prev:tuple, losted: dict):
    losts = ids_prev[0].difference(set(trackers[:, -1]))
    for d in ids_prev[1]:
        if d[-1] in losts:
            losted[d[-1]] = d
    losted = {k: v for k, v in losted.items() if k not in set(trackers[:, -1])}
    for l in losted.values():
        l = l.astype(np.int32)
        mask = str(bin(int(l[6])))[2:].zfill(6)
        for i in range(0, len(l), 2):
            if i > 4: break
            if mask[i] != "1" and i != 1:
                cv2.circle(frame, (l[i], l[i + 1]), 4, (255, 255, 255), 4)
            if i == 4:
                cv2.circle(frame, (l[i], l[i + 1]), 4, (255, 255, 255), 4)
        cv2.putText(frame, f"{l[-1]}(DEAD)", (l[0], l[1]), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame, losted
    
MODE_SAVE = 0
MODE_SHOW = 1

mode = MODE_SHOW

# path_csv = "sources/out_DLC_18fps/v18DLC_dlcrnetms5_bee1011_18Oct11shuffle1_200000_el.csv"
# path_pkl = "sources/out_DLC_18fps/v18DLC_dlcrnetms5_bee1011_18Oct11shuffle1_200000_full.pickle"
path_csv = "/kpsort/sources/out_DLC_18fps_2/v18DLC_dlcrnetms5_1011_18_2Feb26shuffle1_200000_el.csv"
path_pkl = "/kpsort/sources/out_DLC_18fps_2/v18DLC_dlcrnetms5_1011_18_2Feb26shuffle1_200000_full.pickle"

with open(path_pkl, "rb") as file:
    data_pkl: dict = pickle.load(file)
data_csv = load_csv(path_csv)
color_map = iter(gen_random_colors(10000, 334))

model = YOLO("best2.pt", task="predict")
cap = cv2.VideoCapture("sources/v18.mp4")

th = 0.01

fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter(f"output/video_{th}.mp4",fourcc, cap.get(cv2.CAP_PROP_FPS), size)

mot_tracker = Sort(oks_threshold=0.0001)

c = 0
colors = dict()
ids_counter = dict()
prog = tqdm(desc="Generating", total=cap.get(cv2.CAP_PROP_FRAME_COUNT))
losted = dict()
ids_prev = None

while True:
    if c > 500:
        ids_counter = sorted(ids_counter.items())
        ids_counter = dict((x, y) for x, y in ids_counter) 
        plt.bar(ids_counter.keys(), ids_counter.values())
        plt.savefig(f"output/figure/trackrets_{th}.png")
        plt.cla()
        print(len(ids_counter))
        break
    success, frame = cap.read()
    if success:
        individuals, frame_ = assemble_w_yolo(model, frame, data_pkl, data_csv, th, c)
        cv2.putText(frame, str(c), (100, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 1, cv2.LINE_AA)
        desirable2remove = check_overlap_2(individuals, 0.5)
        for individual in individuals:
            for i in range(0, len(individual) - 1, 2):
                if math.isnan(individual[i]): continue
                cv2.circle(frame, (int(individual[i]), int(individual[i + 1])), 5, (0, 255, 0), 5)
        trackers = mot_tracker.update(individuals, desirable2remove, th)
        #print(len(ids_count))
        if c != 0:
            frame, losted = mark_losted_trackers(frame, trackers, ids_prev, losted)

        for d in trackers:
            d = d.astype(np.int32)
            if d[-1] not in colors:
                colors[d[-1]] = next(color_map)
            d_caring = False
            d_exchange = False
            if str(d[-1]) not in ids_counter:
                ids_counter[str(d[-1])] = 1
            else:
                ids_counter[str(d[-1])] += 1
            mask = str(bin(int(d[6])))[2:].zfill(6)
            for i in range(0, len(d), 2):
                if i > 4: break
                if mask[i] != "1" and i != 1:
                    cv2.circle(frame, (d[i], d[i + 1]), 4, colors[d[-1]], 4)
                    #cv2.drawMarker(frame, (d[i], d[i + 1]), colors[d[7]])
                if i == 4:
                    cv2.circle(frame, (d[i], d[i + 1]), 4, colors[d[-1]], 4)
                    #cv2.putText(frame, f"@{hive.pos2id((d[i], d[i + 1]))}", (d[0], d[1]), cv2.FONT_HERSHEY_PLAIN, 5, colors[d[-1]], 1, cv2.LINE_AA)
            cv2.putText(frame, str(d[-1]), (d[0], d[1]), cv2.FONT_HERSHEY_PLAIN, 5, colors[d[-1]], 1, cv2.LINE_AA)
        ids_prev = (set(trackers[:, -1]), trackers)
    
    c += 1
    prog.update(1)
    video.write(frame)
    
    