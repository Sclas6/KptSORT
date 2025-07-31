import os
os.chdir("/kpsort")
import cv2
from tools.kpsort import Sort
from tools.loadpkl_jit import *
from tools.AssignBeeHive import AssignBeeHive
from ultralytics import YOLO
import numpy as np
import pickle
import math
from tqdm import tqdm
from numba import njit
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from collections import deque

"""
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
"""
class Bee():
    hived_series: np.ndarray
    exchanged_series: np.ndarray
    distances: np.ndarray
    def __init__(self, id: int, pos: tuple, length: int=10):
        self.id = id
        self.age = 0
        self.distance = 0
        self.length = length
        self.pos = pos
        #self.trajectory = pl.DataFrame({"y": [pos[0]], "x": [pos[1]]})
        self.trajectory_deque = deque([np.array(pos)], maxlen=length)

        self.tracked_frames = 0
        self.care_frames = 0
        self.feeding_hives = dict()
        self.exchanged_frames = 0
        self.exchanging = dict()
    
    def update(self, pos):
        """
        self.trajectory = pl.concat([self.trajectory, pl.DataFrame({"y": [pos[0]], "x": [pos[1]]})])
        if len(self.trajectory) > self.length:
            self.trajectory = self.trajectory[1:]
        self.age += 1
        self.distance += math.dist(pos, self.pos) / 44
        self.pos = pos
        """
        self.trajectory_deque.append(np.array(pos)) # posをNumPy配列に変換して格納
        self.age += 1
        self.distance = math.dist(pos, self.pos) / 44
        self.pos = pos
        
    def draw_trajectory(self, frame, color=(0,0,255)):
        #cv2.polylines(frame, [self.trajectory.to_numpy()], False, color, 5)
        points = np.array(self.trajectory_deque).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(frame, [points], False, color, 5)

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
            dist = np.linalg.norm(np.array([d[0], d[1]], dtype=np.float32) - np.array([d[4], d[5]], dtype=np.float32))
        elif d[6] == 3:
            dist = np.linalg.norm(np.array([d[0], d[1]], dtype=np.float32) - np.array([d[2], d[3]], dtype=np.float32))
        elif d[6] == 12:
            dist = np.linalg.norm(np.array([d[0], d[1]], dtype=np.float32) - np.array([d[4], d[5]], dtype=np.float32))
        else:
            dist = 0
            count -= 1
        sum += dist
    return sum / count

@njit
def calc_unit_vector(d):
    if d[6] == 0:
        v = np.array([d[0] - d[2], d[1] - d[3]], dtype=np.float32)
        return v / np.linalg.norm(v)
    elif d[6] == 3:
        v = np.array([d[0] - d[2], d[1] - d[3]], dtype=np.float32)
        return v / np.linalg.norm(v)
    elif d[6] == 12:
        v = np.array([d[0] - d[4], d[1] - d[5]], dtype=np.float32)
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
    
def gen_graphs(frames, bees: dict, colors: dict, path_out: str, th_frames: int=125):
    """
    with open(f'{path_out}hived_counter.pkl', mode='wb') as fo:
        pickle.dump(counter.care_counter, fo)
    """
    #print(([str(k) for k, v in counter.cared_counter.items() if v > 10], [v for v in counter.cared_counter.values() if v > 10]))
    #plt.bar([str(k) for k, v in counter.cared_counter.items() if v > 10], [v for v in counter.cared_counter.values() if v > 10])
    cared_counter_sum = dict()
    for id in bees:
        #{hive_id: count, ...}
        for hive_id, count in bees[id].feeding_hives.items():
            if hive_id not in cared_counter_sum.keys():
                cared_counter_sum[hive_id] = 0
            cared_counter_sum[hive_id] += count
    #print(([str(k) for k, v in cared_counter_sum.items() if v != 0], [v for v in cared_counter_sum.values() if v != 0]))
    elements = [k for k, v in cared_counter_sum.items() if v > 0]
    for i, id in enumerate(bees):
        bee: Bee = bees[id]
        if i == 0:
            plt.bar([str(k) for k in bee.feeding_hives.keys() if k in elements], [v for k, v in bee.feeding_hives.items() if k in elements], color=(colors[bee.id][0]/255,colors[bee.id][1]/255,colors[bee.id][2]/255))
        else:
            plt.bar([str(k) for k in bee.feeding_hives.keys() if k in elements], [v for k, v in bee.feeding_hives.items() if k in elements], bottom=[v for k, v in bees[i - 1].feeding_hives.items() if k in elements], color=(colors[bee.id][0]/255,colors[bee.id][1]/255,colors[bee.id][2]/255))
    """
        if i == 0:
            plt.bar([str(k) for k in counter.cared_counter[key].keys() if k in elements], [v for k, v in counter.cared_counter[key].items() if k in elements], color=(colors[key][0]/255,colors[key][1]/255,colors[key][2]/255))
        else:
            plt.bar([str(k) for k in counter.cared_counter[key].keys() if k in elements], [v for k, v in counter.cared_counter[key].items() if k in elements], bottom=[v for k, v in counter.cared_counter[list(counter.cared_counter.keys())[i - 1]].items() if k in elements], color=(colors[key][0]/255,colors[key][1]/255,colors[key][2]/255))
    """
    plt.savefig(f"{path_out}hived_counter.png")
    plt.cla()
    trajectories = {bee.id: bee.distance for bee in bees.values()}
    trajectories = sorted(trajectories.items(), key=lambda x:x[1])
    plt.barh([str(i[0]) for i in trajectories], [i[1] for i in trajectories])
    plt.title(f"総移動距離: {np.sum(Bee.distances)}cm")
    plt.suptitle(f"平均移動距離: {np.mean(Bee.distances[:frames])}cm")
    plt.savefig(f"{path_out}trajectories.png")
    plt.cla()
    plt.suptitle("")

    plt.plot([i for i in range(frames)], Bee.distances[:frames])
    plt.savefig(f"{path_out}trajectories_series.png")
    plt.cla()


    #print(counter.ids_counter)
    exchanged_map = np.zeros((len(bees) + 1, len(bees) + 1))
    for id1 in bees:
        bee1: Bee = bees[id1]
        for id2 in bee1.exchanging:
            if bee1.exchanging[id2] > 0:
                exchanged_map[id1][int(id2)] = bee1.exchanging[int(id2)]

    ids_counter = {str(bee.id): bee.tracked_frames for bee in bees.values() if bee.tracked_frames > th_frames}
    plt.bar(ids_counter.keys(), ids_counter.values())
    plt.savefig(f"{path_out}trackrets.png")
    plt.cla()
    plt.plot([i for i in range(frames)], Bee.hived_series[:frames])
    
    ratio_sum = 0
    for i in ids_counter.keys():
        ratio = ids_counter[i] / frames
        ratio_sum += ratio
    avg_ratio = ratio_sum / len(ids_counter.keys())
    print(f"AVG: {avg_ratio}")
    
    plt.savefig(f"{path_out}hived_series.png")
    plt.cla()
    plt.plot([i for i in range(frames)], Bee.exchanged_series[:frames])
    plt.savefig(f"{path_out}exchanged_series.png")
    plt.cla()
    sns.heatmap(exchanged_map, cmap='Blues')
    plt.savefig(f"{path_out}exchanged_map.png")

#@njit
def detect_trophallaxis(d, trackers, bees: dict, fps=18):
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
                    bees[d[-1]].exchanged_frames += 1
                    if bees[d[-1]].exchanged_frames > fps:
                        d_exchange = True
                        if d2[-1] not in bees[d[-1]].exchanging:
                            bees[d[-1]].exchanging[d2[-1]] = 1
                        else:
                            bees[d[-1]].exchanging[d2[-1]] += 1
    return d_exchange

#@njit
def detect_caring(d, mask, hive: AssignBeeHive, img, bee: Bee, fps=18):
    d_caring = False
    if mask[0] == '1':
        dur = fps * 1
        
        bee.care_frames += 1
        if bee.care_frames > dur:
            """if counter.hived_counter_series[counter.c - 1] == 0:
                for cc in range(int(dur)):
                    if counter.c - int(dur) + cc >= 0:
                        counter.inc(hived_series=(counter.c-int(dur)+cc, 1))"""
            #cv2.putText(frame, "!!!", (d[4], d[5]), cv2.FONT_HERSHEY_PLAIN, 5.0, colors[d[-1]], 5, cv2.LINE_AA)
            d_caring = True
            bee.feeding_hives[hive.pos2id((d[2], d[2 + 1]))] += 1
            #counter.inc(cared=(hive.pos2id((d[2], d[2 + 1])), 1), hived_series=(counter.c, 1))
    else:
        bee.care_frames = 0
        
    return d_caring, frame


MODE_SAVE = 0
MODE_SHOW = 1

mode = MODE_SHOW

path_csv = "sources/DLC/out_DLC_0430/dlc_pos9_resnet50.csv"
path_pkl = "sources/DLC/out_DLC_0430/dlc_pos9_resnet50.pickle"

with open(path_pkl, "rb") as file:
    data_pkl: dict = pickle.load(file)
data_csv = load_csv(path_csv)
color_map = iter(gen_random_colors(10000, 334))

model = YOLO("/kpsort/runs/obb/train5/weights/best.pt")
cap = cv2.VideoCapture("/kpsort/sources/Videos/resized_0430.mp4")

th = 0.2

fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter(f"output/videos/{th}_0703.mp4",fourcc, fps, size)

img_hive_sam = cv2.imread("/kpsort/result/pps64_cnl3_1/result_pps64_cnl3_1.png")

colors = dict()
ids_prev = None
losted = dict()

mot_tracker = Sort(oks_threshold=0.00001)
#counter = Counter()
bees = dict()

with open("hive_2.pkl", "rb") as f:
    hive = pickle.load(f)
    #counter.cared_counter = {h.id: 0 for h in hive.hives}

data_raw = list()
for i, _ in tqdm(enumerate(data_pkl), total=len(data_pkl.keys())):
    if i > 0:
        data_raw.append(pkl2setlist(data_pkl, i - 1))

Bee.exchanged_series = np.zeros((frames))
Bee.hived_series = np.zeros((frames))
Bee.distances = np.zeros((frames))

c = 0
prog = tqdm(desc="Generating", total=frames)
while True:
    success, frame = cap.read()
    
    if c > 250:
        gen_graphs(c, bees, colors, "output/graphs/", 0)
        for bee in bees.values():
            #print(f"{bee.id}: {bee.distance}")
            pass
        #print(counter.cared_counter)
        #gen_graphs(counter, colors, "output/graphs/")
        
        break
    if success:
        results = model.predict(frame, device=0, conf=0.45, verbose=False)
        rects = results[0].obb.xyxyxyxy.to('cpu').detach().numpy().copy()
        individuals = assemble_w_yolo(rects, data_raw[c], data_csv[c], th)
        cv2.putText(frame, str(c), (100, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 1, cv2.LINE_AA)
        desirable2remove = check_overlap_2(individuals, 0.5)
        
        for individual in individuals:
            for i in range(0, len(individual) - 1, 2):
                if math.isnan(individual[i]): continue
                cv2.circle(frame, (int(individual[i]), int(individual[i + 1])), 5, (0, 255, 0), 5)
        trackers = mot_tracker.update(individuals, desirable2remove, th)
        #trackers = []
        """
        if counter.c != 0:
            frame, losted = mark_losted_trackers(frame, trackers, ids_prev, losted)
        """
        pred_ids = [d[-1] for d in trackers]
        #print(([str(k) for k, v in counter.cared_counter.items() if v != 0], [v for v in counter.cared_counter.values() if v != 0]))
                
        for d in trackers:
            d = d.astype(np.int32)
            if d[-1] not in colors:
                colors[d[-1]] = next(color_map)

            if d[-1] not in bees:
                bees[d[-1]] = Bee(id=d[-1], pos=(d[0], d[1]), length=200)
                bees[d[-1]].feeding_hives = {h.id: 0 for h in hive.hives}
            bees[d[-1]].tracked_frames += 1

            mask = str(bin(int(d[6])))[2:].zfill(6)
            d_exchange = detect_trophallaxis(d, trackers, bees, fps)          
            #d_exchange = False 
            d_caring, _ = detect_caring(d, mask, hive, img_hive_sam, bees[d[-1]], fps)
            #print([v for v in counter.cared_counter[d[-1]].values() if v != 0])
            #d_caring = False

            for i in range(0, len(d[:3 * 2 + 1]), 2):
                if i > 4: break
                if mask[i] != "1":
                    cv2.circle(frame, (d[i], d[i + 1]), 4, colors[d[-1]], 4)
                    """if not hived:
                        cv2.putText(frame, f"@{hive.pos2id((d[i], d[i + 1]))}", (d[0], d[1]), cv2.FONT_HERSHEY_PLAIN, 5, colors[d[7]], 1, cv2.LINE_AA)
                        hived = True
                    if i == 0:
                        cv2.putText(frame, "head", (d[i], d[i + 1]), cv2.FONT_HERSHEY_PLAIN, 3, colors[d[7]], 1, cv2.LINE_AA)
                    if i == 2:
                        cv2.putText(frame, "onaka", (d[i], d[i + 1]), cv2.FONT_HERSHEY_PLAIN, 3, colors[d[7]], 1, cv2.LINE_AA)
                    if i == 4:
                        cv2.putText(frame, "Sting", (d[i], d[i + 1]), cv2.FONT_HERSHEY_PLAIN, 3, colors[d[7]], 1, cv2.LINE_AA)"""
            cv2.putText(frame, str(d[-1]), (d[0], d[1]), cv2.FONT_HERSHEY_PLAIN, 5, colors[d[-1]], 1, cv2.LINE_AA)
            if d_caring:
                cv2.circle(frame, (d[4], d[5]), 10, (0, 0, 255), 10)
                Bee.hived_series[c] += 1
            if d_exchange:
                cv2.circle(frame, (d[0], d[1]), 10, (0, 255, 0), 10)
                Bee.exchanged_series[c] += 1
            else:
                bees[d[-1]].update(pos=(d[0], d[1]))
                bees[d[-1]].draw_trajectory(frame, colors[d[-1]])
            Bee.distances[c] += bees[d[-1]].distance
        """
        print(np.sum(Bee.exchanged_series))  
        print(np.sum(Bee.hived_series))
        print(np.sum(Bee.distances))
        print()
        """
        #counter.trajectories = {k: v.distance for k, v in bees.items()}
        #ids_prev = (set(trackers[:, -1]), trackers)
                
        c += 1
        prog.update(1)
        video.write(frame)
    else: 
        gen_graphs(c, bees, colors, "output/graphs/", 5)
        break