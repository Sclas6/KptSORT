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

def check_overlap(individuals, threshold: int):
    #(individuals)
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
    
MODE_SAVE = 0
MODE_SHOW = 1

mode = MODE_SHOW

path_csv = "sources/out_DLC_18fps/v18DLC_dlcrnetms5_bee1011_18Oct11shuffle1_200000_el.csv"
path_pkl = "sources/out_DLC_18fps/v18DLC_dlcrnetms5_bee1011_18Oct11shuffle1_200000_full.pickle"

with open(path_pkl, "rb") as file:
    data_pkl: dict = pickle.load(file)
data_csv = load_csv(path_csv)
color_map = iter(gen_random_colors(10000, 334))

model = YOLO("best2.pt", task="predict")
cap = cv2.VideoCapture("sources/v18.mp4")

th = 0.2   

fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter(f"output/video_{th}.mp4",fourcc, cap.get(cv2.CAP_PROP_FPS), size)

mot_tracker = Sort(iou_threshold=0.0001)

c = 0
colors = dict()
ids = dict()
prog = tqdm(desc="Generating", total=cap.get(cv2.CAP_PROP_FRAME_COUNT))
while True:
    if c > 500:
        ids = sorted(ids.items())
        ids = dict((x, y) for x, y in ids) 
        plt.bar(ids.keys(), ids.values())
        plt.savefig(f"output/figure/trackrets_{th}.png")
        plt.cla()
        print(len(ids))
        break
    success, frame = cap.read()
    if success:
        individuals, frame_ = assemble_w_yolo(model, frame, data_pkl, data_csv, th, c)
        cv2.putText(frame, str(c), (100, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 1, cv2.LINE_AA)
        tmp = len(individuals)
        desirable2remove = check_overlap(individuals, th)
        for individual in individuals:
            for i in range(0, len(individual) - 1, 2):
                if math.isnan(individual[i]): continue
                cv2.circle(frame, (int(individual[i]), int(individual[i + 1])), 5, (0, 255, 0), 5)
        trackers = mot_tracker.update(individuals, desirable2remove, th)
        print(c)
        for d in trackers:
            print(d)
            d = d.astype(np.int32)
            if d[-1] not in colors:
                colors[d[-1]] = next(color_map)
            d_caring = False
            d_exchange = False
            if str(d[-1]) not in ids:
                ids[str(d[-1])] = 1
            else:
                ids[str(d[-1])] += 1
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
    c += 1
    prog.update(1)
    video.write(frame)
    
    