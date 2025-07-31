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
import pandas as pd
from tqdm import tqdm
from numba import njit
import matplotlib.pyplot as plt
import seaborn as sns

class Counter():
    def __init__(self):
        self.c = 0
        self.ids_counter = dict()
        self.care_counter = dict()
        self.cared_counter = dict()
        self.exchanged_counter = dict()
        self.exchanged_w_id = dict()
        self.hived_counter_series = dict()
        self.exchanged_counter_series = dict()
        self.trajectories = dict()
        
    def update(self, c=None, ids=None, care=None, cared=None, exchanged=None, exchanged_w_id=None, hived_series=None, exchanged_series=None):
        if c != None: self.c = c
        if ids != None:
            if type(ids) == tuple:
                self.ids_counter[ids[0]] = ids[1]
            else:
                self.ids_counter = ids
        if care != None: self.care_counter[care[0]] = care[1]
        if cared != None: self.cared_counter[cared[0]] = cared[1]
        if exchanged != None: self.exchanged_counter[exchanged[0]] = exchanged[1]
        if exchanged_w_id != None:
            if len(exchanged_w_id) == 2:
                self.exchanged_w_id[exchanged_w_id[0]] = exchanged_w_id[1]
            elif len(exchanged_w_id) == 3:
                self.exchanged_w_id[exchanged_w_id[0]][exchanged_w_id[1]] = exchanged_w_id[2]
        if hived_series != None: self.hived_counter_series[hived_series[0]] = hived_series[1]
        if exchanged_series != None: self.exchanged_counter_series[exchanged_series[0]] = exchanged_series[1]
        
    def inc(self, c=None, ids=None, care=None, cared=None, exchanged=None, exchanged_w_id=None, hived_series=None, exchanged_series=None):
        if c != None: self.c += c
        if ids != None: self.ids_counter[ids[0]] += ids[1]
        if care != None: self.care_counter[care[0]] += care[1]
        if cared != None: self.cared_counter[cared[0]] += cared[1]
        if exchanged != None: self.exchanged_counter[exchanged[0]] += exchanged[1]
        if exchanged_w_id != None:
            if len(exchanged_w_id) == 2:
                self.exchanged_w_id[exchanged_w_id[0]] += exchanged_w_id[1]
            elif len(exchanged_w_id) == 3:
                self.exchanged_w_id[exchanged_w_id[0]][exchanged_w_id[1]] += exchanged_w_id[2]
        if hived_series != None: self.hived_counter_series[hived_series[0]] += hived_series[1]
        if exchanged_series != None: self.exchanged_counter_series[exchanged_series[0]] += exchanged_series[1]
        
class Bee():
    def __init__(self, id: int, pos: tuple, length: int=10):
        self.id = id
        self.age = 0
        self.distance = 0
        self.length = length
        self.pos = pos
        self.trajectory = pd.DataFrame({"y": [pos[0]], "x": [pos[1]]})
    
    def update(self, pos):
        self.trajectory = pd.concat([self.trajectory, pd.DataFrame({"y": [pos[0]], "x": [pos[1]]})], axis=0)
        self.trajectory = self.trajectory.reset_index(drop=True)
        if len(self.trajectory) > self.length:
            self.trajectory = self.trajectory.drop(index=0)
        self.trajectory = self.trajectory.reset_index(drop=True)
        self.age += 1
        self.distance += math.dist(pos, self.pos)
        self.pos = pos
        
    def draw_trajectory(self, frame, color=(0,0,255)):
        for i, row in self.trajectory.iloc[::-1].iterrows():
            if i != 0:
                pt1 = (row["y"], row["x"])
                pt2 = (self.trajectory.at[i-1, "y"], self.trajectory.at[i-1, "x"])
                cv2.line(frame, pt1, pt2, color, 5)
                #print(f"draw: {pt1} to {pt2}")
    
        
class Score():
    def __init__(self):
        self.fp = 0
        self.misses = 0
        self.idsw = 0
        self.pre_ids = []
        self.g = 0
        
    def update(self, fp=None, misses=None, idsw=None, pre_ids=None, g=None):
        if fp!=None: self.fp = fp
        if misses!=None: self.misses = misses
        if idsw!=None: self.idsw = idsw
        if pre_ids!=None: self.pre_ids = pre_ids
        if g!=None: self.g = g

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


@njit
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

@njit
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
    
def gen_graphs(counter: Counter, score: Score, colors: dict, path_out: str):
    with open(f'{path_out}hived_counter.pkl', mode='wb') as fo:
        pickle.dump(counter.care_counter, fo)
    #print(([str(k) for k, v in counter.cared_counter.items() if v > 10], [v for v in counter.cared_counter.values() if v > 10]))
    #plt.bar([str(k) for k, v in counter.cared_counter.items() if v > 10], [v for v in counter.cared_counter.values() if v > 10])
    cared_counter_sum = dict()
    for v in counter.cared_counter.values():
        #{hive_id: count, ...}
        for hive_id, count in v.items():
            if hive_id not in cared_counter_sum.keys():
                cared_counter_sum[hive_id] = 0
            cared_counter_sum[hive_id] += count
    #print(([str(k) for k, v in cared_counter_sum.items() if v != 0], [v for v in cared_counter_sum.values() if v != 0]))
    elements = [k for k, v in cared_counter_sum.items() if v > 125]
    for i, key in enumerate(counter.cared_counter):
        if i == 0:
            plt.bar([str(k) for k in counter.cared_counter[key].keys() if k in elements], [v for k, v in counter.cared_counter[key].items() if k in elements], color=(colors[key][0]/255,colors[key][1]/255,colors[key][2]/255))
        else:
            plt.bar([str(k) for k in counter.cared_counter[key].keys() if k in elements], [v for k, v in counter.cared_counter[key].items() if k in elements], bottom=[v for k, v in counter.cared_counter[list(counter.cared_counter.keys())[i - 1]].items() if k in elements], color=(colors[key][0]/255,colors[key][1]/255,colors[key][2]/255))
    plt.savefig(f"{path_out}hived_counter.png")
    plt.cla()
    counter.trajectories = sorted(counter.trajectories.items(), key=lambda x:x[1])
    print(counter.trajectories)
    plt.barh([str(i[0]) for i in counter.trajectories], [i[1] for i in counter.trajectories])
    plt.savefig(f"{path_out}trajectories.png")
    plt.cla()
    #print(counter.care_counter)
    exchanged_map = np.zeros((int(max(counter.exchanged_w_id.keys())) + 1, int(max(counter.exchanged_w_id.keys())) + 1))
    for i in counter.exchanged_w_id.keys():
        for j in counter.exchanged_w_id[i]:
            exchanged_map[int(i)][int(j)] = counter.exchanged_w_id[i][j]
    counter.ids_counter = {k: v for k, v in counter.ids_counter.items() if v > 10}
    plt.bar(counter.ids_counter.keys(), counter.ids_counter.values())
    plt.savefig(f"{path_out}trackrets.png")
    plt.cla()
    plt.plot(counter.hived_counter_series.keys(), counter.hived_counter_series.values())
    
    ratio_sum = 0
    for i in counter.ids_counter.keys():
        ratio = counter.ids_counter[i] / counter.c
        ratio_sum += ratio
    avg_ratio = ratio_sum / len(counter.ids_counter.keys())
    print(f"AVG: {avg_ratio}")
    
    plt.savefig(f"{path_out}hived_series.png")
    plt.cla()
    plt.plot(counter.exchanged_counter_series.keys(), counter.exchanged_counter_series.values())
    plt.savefig(f"{path_out}exchanged_series.png")
    plt.cla()
    sns.heatmap(exchanged_map, cmap='Blues')
    plt.savefig(f"{path_out}exchanged_map.png")
    mota = 1 - (score.fp + score.misses + score.idsw) / score.g
    print(f"MISSES: {score.misses}")
    print(f"MOTA: {mota}")
    
def detect_trophallaxis(d, trackers, counter: Counter, fps=18):
    d_exchange = False
    if d[6] in [0, 3, 12]:
        d_head = np.array([d[0], d[1]])
        for d2 in trackers:
            if d2[6] in [0, 3, 12] and d[-1] != d2[-1]:
                d2_head = np.array([d2[0], d2[1]])
                r = np.linalg.norm(d2_head - d_head)
                rad = np.linalg.norm(calc_unit_vector(d) + calc_unit_vector(d2))
                # !! MAGIC NUMBER 1.1, 1.5
                if r < (calc_ava_length(trackers) / 1.1) and rad < 1.5:
                    if str(d[-1]) not in counter.exchanged_counter:
                        counter.update(exchanged=(str(d[-1]), 1))
                    else:
                        counter.inc(exchanged=(str(d[-1]), 1))
                    if counter.exchanged_counter[str(d[-1])] > fps:
                        d_exchange = True
                        counter.inc(exchanged_series=(counter.c, 1))
                        if d2[-1] not in counter.exchanged_w_id[d[-1]]:
                            counter.update(exchanged_w_id=(d[-1], d2[-1], 1))
                        else:
                            counter.inc(exchanged_w_id=(d[-1], d2[-1], 1))
    return d_exchange

def detect_caring(d, frame, mask, hive: AssignBeeHive, img, counter: Counter, fps=18):
    d_caring = False
    if mask[0] == '1':
        dur = fps * 5
        
        if d[-1] not in counter.care_counter:
            counter.update(care=(d[-1], 1))
        else:
            counter.inc(care=(d[-1], 1))
            cv2.putText(frame, f"@{hive.pos2id((d[2], d[2 + 1]), img)}", (d[0], d[1]), cv2.FONT_HERSHEY_PLAIN, 5, colors[d[-1]], 1, cv2.LINE_AA)
        if counter.care_counter[d[-1]] > dur:
            """if counter.hived_counter_series[counter.c - 1] == 0:
                for cc in range(int(dur)):
                    if counter.c - int(dur) + cc >= 0:
                        counter.inc(hived_series=(counter.c-int(dur)+cc, 1))"""
            #cv2.putText(frame, "!!!", (d[4], d[5]), cv2.FONT_HERSHEY_PLAIN, 5.0, colors[d[-1]], 5, cv2.LINE_AA)
            d_caring = True
            counter.cared_counter[d[-1]][hive.pos2id((d[2], d[2 + 1]))] += 1
            #counter.inc(cared=(hive.pos2id((d[2], d[2 + 1])), 1), hived_series=(counter.c, 1))
    else:
        if str(d[-1]) in counter.care_counter:
            counter.update(care=(d[-1], 0))
    return d_caring, frame


MODE_SAVE = 0
MODE_SHOW = 1

mode = MODE_SHOW

path_csv = "sources/DLC/out_DLC_0430/dlc_pos9_resnet50.csv"
path_pkl = "sources/DLC/out_DLC_0430/dlc_pos9_resnet50.pickle"

#path_csv = "/kpsort/sources/DLC/out_DLC_30fps/resized_IMG_7730_30fpsDLC_dlcrnetms5_bee1011_18Oct17shuffle1_200000_el.csv"
#path_pkl = "/kpsort/sources/DLC/out_DLC_30fps/resized_IMG_7730_30fpsDLC_dlcrnetms5_bee1011_18Oct17shuffle1_200000_full.pickle"


with open(path_pkl, "rb") as file:
    data_pkl: dict = pickle.load(file)
data_csv = load_csv(path_csv)
color_map = iter(gen_random_colors(10000, 334))

model = YOLO("/kpsort/runs/obb/train5/weights/best.pt")
#model = YOLO("sources/Models/bench_obb.pt")
cap = cv2.VideoCapture("/kpsort/sources/Videos/resized_0430.mp4")
#cap = cv2.VideoCapture("/kpsort/sources/Videos/resized_IMG_7730_30fps.mp4")


th = 0.2

fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter(f"output/videos/{th}_0703.mp4",fourcc, fps, size)

colors = dict()
ids_prev = None
losted = dict()

mot_tracker = Sort(oks_threshold=0.00001)
counter = Counter()
score = Score()
bees = dict()

with open("hive_2.pkl", "rb") as f:
    hive = pickle.load(f)
    #counter.cared_counter = {h.id: 0 for h in hive.hives}
    
data_raw = list()
for i, k in tqdm(enumerate(data_pkl), total=len(data_pkl.keys())):
    if i > 0:
        data_raw.append(pkl2setlist(data_pkl, i - 1))

prog = tqdm(desc="Generating", total=cap.get(cv2.CAP_PROP_FRAME_COUNT))
while True:
    success, frame = cap.read()
    counter.update(hived_series=(counter.c, 0), exchanged_series=(counter.c, 0))
    
    if counter.c > 200:
        counter.update(ids=dict((x, y) for x, y in sorted(counter.ids_counter.items())))
        plt.bar(counter.ids_counter.keys(), counter.ids_counter.values())
        plt.savefig(f"output/figure/trackrets_{th}.png")
        plt.cla()
        gen_graphs(counter, score, colors, "output/graphs/")
        for bee in bees.values():
            print(f"{bee.id}: {bee.distance}")
        
        break
    if success:
        individuals, _ = assemble_w_yolo(model, frame, data_raw[counter.c], data_csv[counter.c], th)
        cv2.putText(frame, str(counter.c), (100, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 1, cv2.LINE_AA)
        desirable2remove = check_overlap_2(individuals, 0.5)
            
        trackers = mot_tracker.update(individuals, desirable2remove, th)
        #if counter.c != 0:
            #frame, losted = mark_losted_trackers(frame, trackers, ids_prev, losted)

        pred_ids = [d[-1] for d in trackers]
        score.update(misses=score.misses + len(set(score.pre_ids) - set(pred_ids)) if counter.c != 0 else 0, g=score.g + 10, pre_ids=pred_ids)
        for d in trackers:
            if d[-1] not in counter.exchanged_w_id:
                counter.update(exchanged_w_id=(d[-1], dict()))
                
        for d in trackers:
            d = d.astype(np.int32)
            if d[-1] not in colors:
                colors[d[-1]] = next(color_map)
            if str(d[-1]) not in counter.ids_counter:
                counter.update(ids=(str(d[-1]), 1))
            else:
                counter.inc(ids=(str(d[-1]), 1))
            mask = str(bin(int(d[6])))[2:].zfill(6)
            if d[-1] not in counter.cared_counter.keys():
                counter.cared_counter[d[-1]] = {h.id: 0 for h in hive.hives}

            for i in range(0, len(d[:3 * 2 + 1]), 2):
                if i > 4: break
                if mask[i] != "1":
                    cv2.circle(frame, (d[i], d[i + 1]), 10, colors[d[-1]], 5)
            cv2.putText(frame, str(d[-1]), (d[0], d[1]), cv2.FONT_HERSHEY_PLAIN, 5, colors[d[-1]], 3, cv2.LINE_AA)
            if d[-1] not in bees:
                bees[d[-1]] = Bee(id=d[-1], pos=(d[0], d[1]), length=200)
            else:
                bees[d[-1]].update(pos=(d[0], d[1]))
                bees[d[-1]].draw_trajectory(frame, colors[d[-1]])
        
        counter.trajectories = {k: v.distance for k, v in bees.items()}

        ids_prev = (set(trackers[:, -1]), trackers)
                
        counter.inc(c=1)
        prog.update(1)
        video.write(frame)
    else: 
        score.update(idsw=15)
        gen_graphs(counter, score, colors, "output/graphs/")
        break