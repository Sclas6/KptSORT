import os
os.chdir("/kpsort")
import cv2
from tools.kpsort import Sort
from tools.loadpkl import *
from tools.AssignBeeHive import AssignBeeHive
from ultralytics import YOLO
import numpy as np
import pickle
import math
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

def check_overlap_2(individuals, threshold: int):
    fulls_sorted = list()
    fulls = list()
    for individual in individuals:
        if np.any(np.isnan(individual)):
            pass
        else:
            tmp = list()
            for i in range(0, len(individual) - 1, 2):
                tmp.append((individual[i], individual[i + 1]))
            tmp = sorted(tmp)
            fulls_sorted.append(np.append(np.array(tmp).flatten(), 0))
            fulls.append(individual)
    removes = set()
    print(fulls)
    print(fulls_sorted)
    for i, individual in enumerate(fulls_sorted):
        for j, ind in enumerate(fulls_sorted):
            if i == j: continue
            oks_value = oks(individual, ind, 0.1)
            if oks_value > threshold:
                removes.add(i)
    removes = list(removes)
    if len(removes) != 0:
        individuals = np.delete(individuals, removes, 0)
    return individuals

def calc_ava_length(trackers):
    sum = 0
    count = len(trackers)
    for d in trackers:
        if d[6] == 0:
            dist = np.linalg.norm(np.array(d[0], d[1]) - np.array(d[4], d[5]))
        elif d[6] == 3:
            dist = np.linalg.norm(np.array(d[0], d[1]) - np.array(d[2], d[3]))
        elif d[6] == 12:
            dist = np.linalg.norm(np.array(d[0], d[1]) - np.array(d[4], d[5]))
        else:
            dist = 0
            count -= 1
        sum += dist
    return sum / count

def calc_unit_vector(d):
    if d[6] == 0:
        v = np.array([d[0] - d[2], d[1] - d[3]])
        return v / np.linalg.norm(v)
    elif d[6] == 3:
        v = np.array([d[0] - d[2], d[1] - d[3]])
        return v / np.linalg.norm(v)
    elif d[6] == 12:
        v = np.array([d[0] - d[4], d[1] - d[5]])
        return v / np.linalg.norm(v)
    else: return None
    
def detect_trophallaxis():
    pass

ids = dict()

with open("hive.pkl", "rb") as f:
    hive = pickle.load(f)
    hived_counter = {h.id: 0 for h in hive.hives}
    #print(hived_counter)
    
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

fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter("output/video12.mp4",fourcc, cap.get(cv2.CAP_PROP_FPS), size)

mot_tracker = Sort(iou_threshold=0.00001)

c = 0
colors = dict()
hived = dict()
exchanged = dict()
exchanged_w_id = dict()

hived_series = dict()
exchanged_series = dict()

fp = 0
misses = 0
idsw = 0
pre_ids = []
g = 0

prog = tqdm(desc="Generating", total=cap.get(cv2.CAP_PROP_FRAME_COUNT))
while True:
    success, frame = cap.read()
    hived_series[c] = 0
    exchanged_series[c] = 0
    #if c > 500: break
    if c > 500:
        for k in hived_counter:
            if hived_counter[k] != 0:
                print(hived_counter[k])
        break
    if success:
        individuals, frame_ = assemble_w_yolo(model, frame, data_pkl, data_csv, c)
        individuals = check_overlap_2(individuals, 0.4)
        for individual in individuals:
            for i in range(0, len(individual) - 1, 2):
                if math.isnan(individual[i]): continue
                #cv2.circle(frame, (int(individual[i]), int(individual[i + 1])), 7, (255, 255, 255), 7)
        trackers = mot_tracker.update(individuals)
        length_ava = calc_ava_length(trackers)
        pred_ids = [d[-1] for d in trackers]
        misses += len(set(pre_ids) - set(pred_ids)) if c != 0 else 0
        g += 10
        pre_ids = pred_ids
        for d in trackers:
            if d[-1] not in exchanged_w_id:
                exchanged_w_id[d[-1]] = dict()
        for d in trackers:
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
            
            # Detect Eiyou
            if d[6] in [0, 3, 12]:
                d_head = np.array([d[0], d[1]])
                for d2 in trackers:
                    if d2[6] in [0, 3, 12] and d[-1] != d2[-1]:
                        d2_head = np.array([d2[0], d2[1]])
                        r = np.linalg.norm(d2_head - d_head)
                        rad = np.linalg.norm(calc_unit_vector(d) + calc_unit_vector(d2))
                        # !! MAGIC NUMBER 1.1, 1.5
                        if r < (length_ava / 1.1) and rad < 1.5:
                            if str(d[-1]) not in exchanged:
                                exchanged[str(d[-1])] = 1
                            else:
                                exchanged[str(d[-1])] += 1
                            if exchanged[str(d[-1])] > cap.get(cv2.CAP_PROP_FPS):
                                d_exchange = True
                                exchanged_series[c] += 1
                                if d2[-1] not in exchanged_w_id[d[-1]]:
                                    exchanged_w_id[d[-1]][d2[-1]] = 1
                                else:
                                    exchanged_w_id[d[-1]][d2[-1]] += 1
                            
            # Detect Caring
            if mask[0] == '1':
                dur = cap.get(cv2.CAP_PROP_FPS) * 5
                if str(d[-1]) not in hived:
                    hived[str(d[-1])] = 1
                else:
                    hived[str(d[-1])] += 1
                    cv2.putText(frame, f"@{hive.pos2id((d[2], d[2 + 1]))}", (d[0], d[1]), cv2.FONT_HERSHEY_PLAIN, 5, colors[d[-1]], 1, cv2.LINE_AA)
                if hived[str(d[-1])] > dur:
                    if hived_series[c - 1] == 0:
                        for cc in range(int(dur)):
                            if c - int(dur) + cc >= 0:
                                hived_series[c - int(dur) + cc] += 1
                                #hived_counter[hive.pos2id((d[i], d[i + 1]))] += 1
                    #cv2.putText(frame, "!!!", (d[4], d[5]), cv2.FONT_HERSHEY_PLAIN, 5.0, colors[d[-1]], 5, cv2.LINE_AA)
                    d_caring = True
                    hived_counter[hive.pos2id((d[2], d[2 + 1]))] += 1
                    #print(hive.pos2id((d[0], d[1])))
                    #print(d)
                    hived_series[c] += 1
            else:
                if str(d[-1]) in hived:
                    hived[str(d[-1])] = 0
            for i in range(0, len(d[:3 * 2 + 1]), 2):
                if i > 4: break
                if mask[i] != "1":
                    cv2.circle(frame, (d[i], d[i + 1]), 5, colors[d[-1]], 5)
                    #cv2.drawMarker(frame, (d[i], d[i + 1]), colors[d[7]])
                    """if not hived:
                        cv2.putText(frame, f"@{hive.pos2id((d[i], d[i + 1]))}", (d[0], d[1]), cv2.FONT_HERSHEY_PLAIN, 5, colors[d[7]], 1, cv2.LINE_AA)
                        hived = True
                    if i == 0:
                        cv2.putText(frame, "head", (d[i], d[i + 1]), cv2.FONT_HERSHEY_PLAIN, 3, colors[d[7]], 1, cv2.LINE_AA)
                    if i == 2:
                        cv2.putText(frame, "onaka", (d[i], d[i + 1]), cv2.FONT_HERSHEY_PLAIN, 3, colors[d[7]], 1, cv2.LINE_AA)
                    if i == 4:
                        cv2.putText(frame, "Sting", (d[i], d[i + 1]), cv2.FONT_HERSHEY_PLAIN, 3, colors[d[7]], 1, cv2.LINE_AA)"""
            cv2.putText(frame, str(d[-1]), (d[0], d[1]), cv2.FONT_HERSHEY_PLAIN, 4.0, colors[d[-1]], 1, cv2.LINE_AA)
            if d_caring:
                cv2.circle(frame, (d[4], d[5]), 10, (0, 0, 255), 10)
            if d_exchange:
                cv2.circle(frame, (d[0], d[1]), 10, (0, 255, 0), 10)
                        
        c += 1
        prog.update(1)
        video.write(frame)
    else: 
        with open('hived_counter.pkl', mode='wb') as fo:
            pickle.dump(hived_counter, fo)
        exchanged_map = np.zeros((int(max(exchanged_w_id.keys())) + 1, int(max(exchanged_w_id.keys())) + 1))
        for i in exchanged_w_id.keys():
            for j in exchanged_w_id[i]:
                exchanged_map[int(i)][int(j)] = exchanged_w_id[i][j]
        plt.bar(ids.keys(), ids.values())
        plt.savefig("trackrets.png")
        plt.cla()
        plt.plot(hived_series.keys(), hived_series.values())
        
        ratio_sum = 0
        for i in ids.keys():
            ratio = ids[i] / cap.get(cv2.CAP_PROP_FRAME_COUNT)
            ratio_sum += ratio
        avg_ratio = ratio_sum / len(ids.keys())
        print(avg_ratio)
        
        plt.savefig("hived_series.png")
        plt.cla()
        plt.plot(exchanged_series.keys(), exchanged_series.values())
        plt.savefig("exchanged_series.png")
        plt.cla()
        sns.heatmap(exchanged_map, cmap='Blues')
        plt.savefig("exchanged_map.png")
        #print(exchanged_map)
        mota = 1 - (fp + misses + idsw)/ g
        print(misses)
        print(mota)
        break