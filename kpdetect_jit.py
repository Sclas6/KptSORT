import os
os.chdir("/kpsort")
from collections import deque
from dataclasses import dataclass
from tools.kpsort import Sort
from tools.loadpkl_jit import *
from tools.AssignBeeHive import AssignBeeHive, Hive
from ultralytics import YOLO
from tqdm import tqdm
from tqdm.contrib import tzip
from numba import njit
import cv2
import collections
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

@dataclass
class CaringEvent:
    id_hive:    int
    duration:   int

class Bee():
    hived_series: np.ndarray
    exchanged_series: np.ndarray
    distances_avg: np.ndarray
    distances_med: np.ndarray
    def __init__(self, id: int, pos: tuple, length: int=10):
        self.id = id
        self.age = 0
        self.distance = 0
        self.distance_sum = 0
        self.length = length
        self.pos = pos
        #self.trajectory = pl.DataFrame({"y": [pos[0]], "x": [pos[1]]})
        self.trajectory_deque = deque([np.array(pos)], maxlen=length)

        self.tracked_frames = 0
        self.feeding_hives = dict()
        self.exchanged_frames = 0
        self.exchanging = dict()
        
        self.care_frames = 0
        self.noncare_frames = 0
        self.care_hives = list()
        self.event_caring = list()
    
    def update(self, pos, fps, reset=False):
        self.trajectory_deque.append(np.array(pos))
        self.age += 1
        if reset:
            self.trajectory_deque = deque([np.array(pos)], maxlen=self.length)
            self.age = 1
            self.pos = pos
            #self.distance = 0
            #self.distance_sum = 0
        if self.age % int(fps * 2) == 0:
            self.distance = math.dist(pos, self.pos) / 44
            self.distance_sum += self.distance
            self.pos = pos
        
    def draw_trajectory(self, frame, img_tracklets, color=(0,0,255)):
        points = np.array(self.trajectory_deque).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(frame, [points], False, color, 5)
        cv2.polylines(img_tracklets, [points], False, color, 5)

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

def gen_hive_heatmap(hive: AssignBeeHive, img_w_sum, path_out):
    @njit(cache=True)
    def _replace_color_numba(img_data, target_color, new_color, alpha):
        height, width, _ = img_data.shape
        output_img = img_data.copy()

        for y in range(height):
            for x in range(width):
                b, g, r, _ = img_data[y, x]
                
                if (b == target_color[0] and 
                    g == target_color[1] and 
                    r == target_color[2]):
                    
                    output_img[y, x] = np.append(new_color, alpha)
                    
        return output_img
    
    img = cv2.cvtColor(img_w_sum, cv2.COLOR_RGB2RGBA)
    img = _replace_color_numba(img, np.array([0,0,0], np.uint8), np.array([0,0,255], np.uint8), 0)

    cm_name = 'jet'
    cm = plt.get_cmap(cm_name)
    h:Hive
    # TODO excute 0 value before clipping
    counter = np.zeros((len(hive.hives), ))
    for i, h in enumerate(hive.hives.values()):
        counter[i] = h.counter
    lower_bound = np.percentile(counter, 5)
    upper_bound = np.percentile(counter, 95)
    if lower_bound != upper_bound:
        counter = np.clip(counter, a_min=lower_bound, a_max=upper_bound)
    hived_max = np.max(counter)
    hived_min = np.min(counter)
    if hived_max == hived_min:
        print("error")
        return
    for c, h in tzip(counter, hive.hives.values()):
        if c == hived_min:
            img = _replace_color_numba(img, np.array(h.color, np.uint8), np.array([0, 0, 0], np.int16), 0)
        else:
            data_normalized = (c - hived_min) / (hived_max - hived_min)
            data_normalized = int(data_normalized * 255)
            rgb = np.array(cm(data_normalized)) * 255
            rgb = rgb.astype(np.int16)
            img = _replace_color_numba(img, np.array(h.color, np.uint8), np.array(rgb[[2, 1, 0]], np.int16), 255)
    
    cv2.imwrite(f"{path_out}hive_heatmap.png", img)
    #return img   
    
def gen_graphs(frames, bees: dict, colors: dict, path_out: str, th_frames: int=125):
    for bee in bees.values():
        e: CaringEvent
        for e in bee.event_caring:
            if e.duration > 1:
                print(e)
        print()
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
            plt.bar([str(k) for k in bee.feeding_hives.keys() if k in elements], [v for k, v in bee.feeding_hives.items() if k in elements], bottom=[v for k, v in bees[id_prev].feeding_hives.items() if k in elements], color=(colors[bee.id][0]/255,colors[bee.id][1]/255,colors[bee.id][2]/255))
        id_prev = id
    plt.title("巣穴の被給餌回数")
    plt.xlabel("巣穴ID")
    plt.xticks(rotation=90)
    plt.ylabel("被給餌回数")
    plt.savefig(f"{path_out}hived_counter.png")
    plt.cla()
    
    trajectories = {bee.id: bee.distance_sum for bee in bees.values()}
    trajectories = sorted(trajectories.items(), key=lambda x:x[1])
    plt.barh([str(i[0]) for i in trajectories if i[1] > 10], [i[1] for i in trajectories if i[1] > 10])
    plt.title(f"総移動距離: {np.sum([i[1] for i in trajectories if i[1] > 10])}cm")
    plt.suptitle(f"平均移動距離: {np.mean([i[1] for i in trajectories if i[1] > 10])}cm")
    plt.xlabel("移動距離(cm)")
    plt.ylabel("個体ID")
    plt.savefig(f"{path_out}trajectories.png")
    plt.cla()
    plt.suptitle("")
    plt.figure(figsize=(80, 30))
    plt.plot([i for i in range(frames)], Bee.distances_avg[:frames])
    with open(f"{path_out}trajectories_series.pkl", mode="wb") as fo:
        pickle.dump([[i for i in range(frames)], Bee.distances_avg[:frames]], fo)
    plt.suptitle("総移動距離の平均値推移", fontsize=60)
    plt.title(f"平均: {np.mean(Bee.distances_avg[:frames])}cm", fontsize=46)
    plt.xlabel("フレーム")
    plt.ylabel("平均移動距離")
    plt.savefig(f"{path_out}trajectories_series.png")

    plt.cla()
    plt.suptitle("")
    plt.figure(figsize=(80, 30))
    plt.plot([i for i in range(frames)], Bee.distances_med[:frames])
    with open(f"{path_out}trajectories_med_series.pkl", mode="wb") as fo:
        pickle.dump([[i for i in range(frames)], Bee.distances_med[:frames]], fo)
    plt.suptitle("総移動距離の平均値推移", fontsize=60)
    plt.title(f"平均: {np.mean(Bee.distances_med[:frames])}cm", fontsize=46)
    plt.xlabel("フレーム")
    plt.ylabel("移動距離の中央値")
    plt.savefig(f"{path_out}trajectories_med_series.png")

    plt.cla()
    plt.suptitle("")
    plt.figure(figsize=(16, 6))

    #print(counter.ids_counter)
    exchanged_map = np.zeros((len(bees) + 1, len(bees) + 1))
    try:
        for id1 in bees:
            bee1: Bee = bees[id1]
            for id2 in bee1.exchanging:
                if bee1.exchanging[id2] > 0:
                    exchanged_map[id1][int(id2)] = bee1.exchanging[int(id2)]
    except Exception as _:
        pass

    ids_counter = {str(bee.id): bee.tracked_frames for bee in bees.values() if bee.tracked_frames > th_frames}
    plt.bar(ids_counter.keys(), ids_counter.values())
    plt.title("継続追跡フレーム数")
    plt.xlabel("個体ID")
    plt.ylabel("フレーム数")
    plt.savefig(f"{path_out}trackrets.png")
    plt.cla()
    
    plt.plot([i for i in range(frames)], Bee.hived_series[:frames])
    
    ratio_sum = 0
    for i in ids_counter.keys():
        ratio = ids_counter[i] / frames
        ratio_sum += ratio
    if len(ids_counter.keys()) != 0:
        avg_ratio = ratio_sum / len(ids_counter.keys())
    else:
        avg_ratio = 0
    print(f"AVG: {avg_ratio}")
    plt.title("フレーム毎給餌回数")
    plt.xlabel("フレーム")
    plt.ylabel("給餌回数")
    plt.savefig(f"{path_out}hived_series.png")

    plt.cla()
    plt.plot([i for i in range(frames)], Bee.exchanged_series[:frames])
    plt.title("フレーム毎栄養交換回数")
    plt.xlabel("フレーム")
    plt.ylabel("栄養交換回数")
    plt.savefig(f"{path_out}exchanged_series.png")

    plt.cla()
    sns.heatmap(exchanged_map, cmap='Blues')
    plt.title("栄養交換回数")
    plt.xlabel("個体ID")
    plt.ylabel("個体ID")
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
    id, dist = hive.pos2id((d[2], d[2 + 1]), img)
    if dist < 50:
        hive.hives[id].counter += 1
    if mask[0] == '1':
        dur = fps
        bee.care_frames += 1
        bee.care_hives.append(id)
        if bee.care_frames > dur:
            d_caring = True
            bee.feeding_hives[id] += 1
    else:
        bee.noncare_frames += 1
        if bee.noncare_frames > fps/2:
            if bee.care_frames != 0:
                bee.event_caring.append(CaringEvent(collections.Counter(bee.care_hives).most_common()[0][0], bee.care_frames))
            bee.noncare_frames = 0
            bee.care_frames = 0
            bee.care_hives.clear()
        
    return d_caring


def kpdetect(filename, hivename, model, n_tracks, n_frames, n_bodyparts=3, th=0.75):    
    if not os.path.exists(f"output/{filename}/"):
        os.makedirs(f"output/{filename}/")    
    def _save(hive, img_hive_sam, c, bees, colors, img_tracklets):
        #print(np.sum(Bee.exchanged_series))
        #print(np.sum(Bee.hived_series))
        gen_hive_heatmap(hive, img_hive_sam, f"output/{filename}/")
        gen_graphs(c, bees, colors, f"output/{filename}/", 100)
        img_tracklets = cv2.cvtColor(img_tracklets, cv2.COLOR_BGR2BGRA)
        img_tracklets[np.all(img_tracklets == [0, 0, 0, 255], axis=2), 3] = 0
        cv2.imwrite(f"output/{filename}/img_tracklets.png", img_tracklets)
    
    path_csv = f"sources/{filename}/CTD.csv"
    path_pkl = f"sources/{filename}/BU.pickle"

    with open(path_pkl, "rb") as file:
        data_pkl: dict = pickle.load(file)
    data_csv = load_csv(path_csv, n_tracks, n_bodyparts)
    color_map = iter(gen_random_colors(10000, 334))

    cap = cv2.VideoCapture(f"sources/{filename}/{filename}.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (width, height)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(frames, n_frames)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video = cv2.VideoWriter(f"output/{filename}/{th}_{filename}_{n_frames}.mp4", fourcc, fps, size)
    print(f"Video\t: {filename}.mp4\nFrames\t: {n_frames}")

    img_hive_sam = cv2.imread(f"sources/hives/{hivename}/result_{hivename}.png")
    img_tracklets = np.zeros((height, width, 3), dtype=np.uint8)
    colors = dict()
    sum_densed = np.zeros((n_frames + 1))

    mot_tracker = Sort(oks_threshold=0.00001, individuals=n_tracks)
    bees = dict()

    hive:AssignBeeHive
    with open(f"sources/hives/{hivename}/{hivename}.pickle", "rb") as f:
        hive = pickle.load(f)
    data_raw = list()
    for i, _ in tqdm(enumerate(data_pkl), total=len(data_pkl.keys())):
        if i > 0:
            data_raw.append(pkl2setlist(data_pkl, i - 1))

    Bee.exchanged_series = np.zeros((frames))
    Bee.hived_series = np.zeros((frames))
    Bee.distances_avg = np.zeros((frames))
    Bee.distances_med = np.zeros((frames))

    c = 0
    prog = tqdm(desc="Generating", total=n_frames)
    while True:
        success, frame = cap.read()
        if c > n_frames:
            _save(hive, img_hive_sam, c, bees, colors, img_tracklets)
            break
        if success:
            
            results = model.predict(frame, device=0, conf=0.45, verbose=False)
            rects = results[0].obb.xyxyxyxy.to('cpu').detach().numpy().copy()
            individuals = assemble_w_yolo(rects, data_raw[c], data_csv[c], th)
            cv2.putText(frame, str(c), (100, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 1, cv2.LINE_AA)
            desirable2remove = check_overlap_2(individuals, 0.5)
            
            poses = np.zeros((0, 2))
            for individual in individuals:
                for i in range(0, len(individual) - 1, 2):
                    if math.isnan(individual[i]): continue
                    cv2.circle(frame, (int(individual[i]), int(individual[i + 1])), 5, (0, 255, 0), 5)
                kpt = np.reshape(individual[:6], (3,2))
                pos_center_2 = np.mean(kpt[~np.isnan(kpt).any(axis=1), :], axis=0)
                poses = np.append(poses, [pos_center_2], axis=0)
            X = StandardScaler().fit_transform(poses)
            
            db = DBSCAN(eps=0.3, min_samples=2)
            db.fit(X)
            labels = db.labels_
            sum_densed[c] = len(labels[labels != -1])

            trackers, respowns = mot_tracker.update(individuals, desirable2remove, th)

            # DENSED INDIVISUALS
            for i in range(max(labels)):
                individuals_labeled = np.array(individuals[np.where(labels == i)])
                for _, point in enumerate(individuals_labeled):
                    kpt = np.reshape(point[:6], (3,2))
            
            for d in trackers:
                d_int = d.astype(np.int32)
                if d_int[-1] not in colors:
                    colors[d_int[-1]] = next(color_map)

                mask = str(bin(int(d_int[6])))[2:].zfill(6)

                kpt = np.reshape(d[:6], (3,2))
                if mask[5] == "1":
                    kpt[2] = np.array([np.nan, np.nan], dtype=np.float32)
                if mask[3] == "1":
                    kpt[1] = np.array([np.nan, np.nan], dtype=np.float32)
                if mask[1] == "1":
                    kpt[0] = np.array([np.nan, np.nan], dtype=np.float32)
                    
                pos_center = np.mean(kpt[~np.isnan(kpt).any(axis=1), :], axis=0)
                if d_int[-1] not in bees:
                    bees[d_int[-1]] = Bee(id=d_int[-1], pos=pos_center, length=200)
                    bees[d_int[-1]].feeding_hives = {h: 0 for h in hive.hives.keys()}
                else:
                    r = d_int[-1] in respowns
                    bees[d_int[-1]].update(pos=pos_center, fps=fps, reset=r)
                    bees[d_int[-1]].draw_trajectory(frame, img_tracklets, colors[d_int[-1]])
                    bees[d_int[-1]].tracked_frames += 1

                d_exchange = detect_trophallaxis(d_int, trackers, bees, fps)          
                d_caring = detect_caring(d_int, mask, hive, img_hive_sam, bees[d_int[-1]], fps)


                for i in range(0, len(d_int[:3 * 2 + 1]), 2):
                    if i > 4: break
                    if mask[i] != "1":
                        cv2.circle(frame, (d_int[i], d_int[i + 1]), 4, colors[d_int[-1]], 4)
                        if i == 0:
                            cv2.circle(frame, (d_int[i], d_int[i + 1]), 8, colors[d_int[-1]], 8)

                cv2.putText(frame, str(d_int[-1]), (d_int[0], d_int[1]), cv2.FONT_HERSHEY_PLAIN, 5, colors[d_int[-1]], 1, cv2.LINE_AA)
                if d_caring:
                    cv2.circle(frame, (d_int[2], d_int[3]), 10, (0, 0, 255), 10)
                    Bee.hived_series[c] += 1
                if d_exchange:
                    cv2.circle(frame, (d_int[0], d_int[1]), 10, (0, 255, 0), 10)
                    Bee.exchanged_series[c] += 1
            #print(bees.keys())
            #print(poses.tolist())
            if len([bees[id].distance for id in bees if id in trackers[:, -1]and bees[id].distance != 0]) != 0 and c != 0:
                Bee.distances_avg[c] = np.mean([bees[id].distance for id in bees if id in trackers[:, -1] and bees[id].distance != 0])
                Bee.distances_med[c] = np.median([bees[id].distance for id in bees if id in trackers[:, -1]and bees[id].distance != 0])
                prog.set_description(f"{np.mean(Bee.distances_avg[:c])*100:.4f} {np.mean(Bee.distances_med[:c])*100:.4f} {np.mean(sum_densed[:c]):.4f}")
            else:
                Bee.distances_avg[c] = 0
                Bee.distances_med[c] = 0
                    
            c += 1
            prog.update(1)
            video.write(frame)
        else: 
            gen_graphs(c, bees, colors, "output/BUCTD/", 100)
            gen_hive_heatmap(hive, img_hive_sam, "output/BUCTD/")
            img_tracklets[np.all(img_tracklets == np.array([0, 0, 0, 255], dtype=np.uint8), axis=2), 3] = 0
            cv2.imwrite("output/BUCTD/img_tracklets.png", img_tracklets)
            break

model = YOLO("/kpsort/runs/obb/train5/weights/best.pt")
kpdetect("resized_0430", "resized_0430", model, 22, 500)
