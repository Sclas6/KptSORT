import os
os.chdir("/kpsort")
from tools.kpsort import Sort
from tools.loadpkl_jit import *
from tools.AssignBeeHive import AssignBeeHive, Hive, Bee, CaringEvent, TrophallaxisEvent
from tools.AssignBeeHive import BEHAVIOR_CARING, BEHAVIOR_NOTHING, BEHAVIOR_TROPHALLAXIS
from ultralytics import YOLO
from tqdm import tqdm
from tqdm.contrib import tzip
from numba import njit
import cv2
import collections
import numpy as np
import pickle
import statistics
from typing import Dict
import itertools
import math
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from scipy.spatial import KDTree

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
    
def save_result(path_out:str, bees: dict):
    with open(f"{path_out}bees.pkl", mode="wb") as f:
        pickle.dump(bees, f)

def gen_graphs(frames, bees: dict, colors: dict, path_out: str, th_frames: int=125):
    """
    with open(f'{path_out}hived_counter.pkl', mode='wb') as fo:
        pickle.dump(counter.care_counter, fo)
    """
    with open(f"{path_out}data_graph.pkl", mode='wb') as fo:
        pickle.dump({"frames": frames, "bees":bees, "colors": colors, "th_frames": th_frames, "hived_series": Bee.hived_series, "exchanged_series": Bee.exchanged_series, "distances_avg": Bee.distances_avg, "distances_med": Bee.distances_med}, fo)
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


def detect_trophallaxis_(d, trackers, bees: dict, frame, fps=18):
    d_exchange = False
    if d[6] in [0, 3, 12]:
        bee: Bee = bees[d[-1]]
        d_head = np.array([d[0], d[1]])
        for d2 in trackers:
            if d2[6] in [0, 3, 12] and d[-1] != d2[-1]:
                d2_head = np.array([d2[0], d2[1]])
                r = np.linalg.norm(d2_head - d_head)
                rad = np.linalg.norm(calc_unit_vector(d) + calc_unit_vector(d2))
                if (r < (calc_ava_length(trackers) / 1.1) and rad < 1.5):
                    if d2[-1] not in bee.trophallaxis_pairs:
                        bee.trophallaxis_pairs[d2[-1]] = 1
                    
                    else:
                        bee.trophallaxis_pairs[d2[-1]] += 1
                #"""
                    if bee.trophallaxis_pairs[d2[-1]] > fps:
                        d_exchange = True
                        bee.update_status(BEHAVIOR_TROPHALLAXIS, frame)
                        if d2[-1] not in bee.exchanging:
                            bee.exchanging[d2[-1]] = 1
                        else:
                            bee.exchanging[d2[-1]] += 1
                #"""
                elif d2[-1] in bee.trophallaxis_pairs:
                    if d2[-1] not in bee.nontrophallaxis_pairs:
                        bee.nontrophallaxis_pairs[d2[-1]] = 1
                    else:
                        bee.nontrophallaxis_pairs[d2[-1]] += 1
                        if bee.nontrophallaxis_pairs[d2[-1]] > fps/2:
                            bee.nontrophallaxis_pairs.pop(d2[-1])
                            bee.event_trophallaxis.append(TrophallaxisEvent(0, int(d2[-1]), bee.trophallaxis_pairs[d2[-1]]))
                        
    return d_exchange

def detect_trophallaxis(bees: dict[int, Bee], trackers, c, scaling_factor, fps=18, radian=1.5, eps=0.02, frame=None):
    start_dur = int(fps) 
    noise_dur = int(fps / 2)
    active_pairs_this_frame = set()
    def _handle_proximity(bee_a: Bee, bee_b: Bee, frame, d_exchanges_ref, index_a):
        bee_a.trophallaxis_pairs[bee_b.id] = bee_a.trophallaxis_pairs.get(bee_b.id, 0) + 1
        bee_a.nontrophallaxis_pairs[bee_b.id] = 0
        if bee_a.trophallaxis_pairs[bee_b.id] > start_dur:
            active_pairs_this_frame.add(tuple(sorted((bee_a.id, bee_b.id))))
            d_exchanges_ref[index_a] = True
            bee_a.update_status(BEHAVIOR_TROPHALLAXIS, frame)
            bee_a.exchanging[bee_b.id] = bee_a.exchanging.get(bee_b.id, 0) + 1
            
            if bee_a.trophallaxis_pairs[bee_b.id] == start_dur + 1:
                # print(f"trophallaxis start: {bee_a.id} with {bee_b.id}")
                for i in range(1, start_dur + 1):
                    bee_a.statuses[frame - i] = BEHAVIOR_TROPHALLAXIS

    def _handle_non_proximity_end(bee_a: Bee, bee_b: Bee, frame):
        if bee_a.trophallaxis_pairs[bee_b.id] > start_dur:
            # print(f"handling... {bee_a.id} & {bee_b.id}")
            bee_a.nontrophallaxis_pairs[bee_b.id] = bee_a.nontrophallaxis_pairs.get(bee_b.id, 0) + 1
            if bee_a.nontrophallaxis_pairs[bee_b.id] <= noise_dur:
                bee_a.update_status(BEHAVIOR_TROPHALLAXIS, frame)
                bee_a.trophallaxis_pairs[bee_b.id] += 1
            else:
                # print(f"trophallaxis end: {bee_a.id} with {bee_b.id}")
                bee_a.event_trophallaxis.append(TrophallaxisEvent(0, int(bee_b.id), bee_a.trophallaxis_pairs[bee_b.id]))
                bee_a.trophallaxis_pairs.pop(bee_b.id)
                bee_a.nontrophallaxis_pairs.pop(bee_b.id)
    
    d_exchanges = np.zeros(len(trackers), bool)
    heads = trackers[:, [0, 1, -1]]
    masks = np.isnan(heads).any(axis=1)
    heads = heads[~np.isnan(heads).any(axis=1), :]
    X = heads[:, :2] * scaling_factor
    db = DBSCAN(eps=eps, min_samples=2)
    db.fit(X)
    labels = (l for l in db.labels_)
    labels = np.array([-1 if b else next(labels) for b in (masks)])
    
    non_negative_mask = (labels != -1)

    original_indices = np.arange(len(labels))
    filtered_values = labels[non_negative_mask]
    mapped_original_indices = original_indices[non_negative_mask]

    unique_values = np.unique(filtered_values)

    for val in unique_values:
        indices_in_filtered = (filtered_values == val).nonzero()[0]
        final_indices = mapped_original_indices[indices_in_filtered]
        for i in itertools.combinations(final_indices, 2):
            if trackers[i[0]][6] in [0, 3, 12] and trackers[i[1]][6] in [0, 3, 12]:
                vec1 = calc_unit_vector(np.ma.filled(trackers[i[0]], fill_value=-1))
                vec2 = calc_unit_vector(np.ma.filled(trackers[i[1]], fill_value=-1))
                rad = np.linalg.norm(vec1 + vec2)
                
                bee1 = bees[trackers[i[0]][-1]]
                bee2 = bees[trackers[i[1]][-1]]
                if rad < radian:
                    _handle_proximity(bee1, bee2, c, d_exchanges, i[0])
                    _handle_proximity(bee2, bee1, c, d_exchanges, i[1])

                #elif bee2.id in bee1.trophallaxis_pairs and bee1.id in bee2.trophallaxis_pairs:
                #    if bee1.id == 12:
                #        print("nontrophallaxis")
                #    _handle_non_proximity_end(bee1, bee2, frame)
                #    _handle_non_proximity_end(bee2, bee1, frame)

    for bee in bees.values():
        ongoing_partners = list(bee.trophallaxis_pairs.keys())
        for partner_id in ongoing_partners:
            if tuple(sorted((bee.id, partner_id))) not in active_pairs_this_frame:
                if partner_id in bees:
                    _handle_non_proximity_end(bee, bees[partner_id], c)
        
                    

    """
    VECTOR_LENGTH_PIXELS = 40 # 描画する矢印の長さ（ピクセル単位）
    VECTOR_COLOR = (0, 0, 255) # 赤 (BGR)
    ARROW_SIZE = 10 # 矢じりの長さ

    for row in trackers:
        if np.any(np.isnan(row)):
            continue
        start_x = int(row[4])
        start_y = int(row[5])
        start_point = (start_x, start_y)
        unit_vector = calc_unit_vector(row)
        if unit_vector is not None:
            end_x = int(start_x + unit_vector[0] * VECTOR_LENGTH_PIXELS)
            end_y = int(start_y + unit_vector[1] * VECTOR_LENGTH_PIXELS)
            end_point = (end_x, end_y)
            cv2.line(frame, start_point, end_point, VECTOR_COLOR, 2)
            angle = np.arctan2(unit_vector[1], unit_vector[0])
            arrow_angle_left = angle - np.pi / 6 
            arrow_left_x = int(end_x - ARROW_SIZE * np.cos(arrow_angle_left))
            arrow_left_y = int(end_y - ARROW_SIZE * np.sin(arrow_angle_left))
            cv2.line(frame, end_point, (arrow_left_x, arrow_left_y), VECTOR_COLOR, 2)
            arrow_angle_right = angle + np.pi / 6
            arrow_right_x = int(end_x - ARROW_SIZE * np.cos(arrow_angle_right))
            arrow_right_y = int(end_y - ARROW_SIZE * np.sin(arrow_angle_right))
            cv2.line(frame, end_point, (arrow_right_x, arrow_right_y), VECTOR_COLOR, 2)
    """
    return {id: b for id, b in zip(trackers[:, -1].astype(int), d_exchanges)}


#@njit
def detect_caring(bee:Bee, hive: AssignBeeHive, img, frame, fps=18):
    d_caring = False
    dur = int(fps/2)
    id, dist = hive.pos2id((bee.kpts[2], bee.kpts[2 + 1]), img)
    l_h2b = np.linalg.norm(bee.kpts[0:2] - [bee.kpts[2:4]])
    l_b2s = np.linalg.norm(bee.kpts[2:4] - [bee.kpts[4:6]])
    body_length = l_h2b + l_b2s

    if len(bee.care_hives) != 0:
        id = collections.Counter(bee.care_hives).most_common()[0][0]
    if dist < 50:
        hive.hives[id].counter += 1

    is_detected = (bee.mask[0] == '1') or (bee.mask[0] == '0' and bee.mask[2] == '0' and l_h2b < (body_length / 4))
    if is_detected:
        bee.care_frames += 1
        bee.care_hives.append(id)
        bee.noncare_frames = 0 #
        
        if bee.care_frames > dur:
            d_caring = True
            bee.feeding_hives[id] += 1
            bee.update_status(BEHAVIOR_CARING, frame)
            if bee.care_frames == dur + 1:
                # print(f"start: {bee.id}")
                for i in range(1, dur + 1):
                    bee.statuses[frame - i] = BEHAVIOR_CARING
    else:
        if bee.care_frames > dur:
            bee.noncare_frames += 1
            if bee.noncare_frames <= dur/2:
                bee.update_status(BEHAVIOR_CARING, frame)
                bee.care_frames += 1
            else:
                # print(f"end: {bee.id}")
                for i in range(1, bee.noncare_frames + 1):
                    bee.statuses[bee.frame_cared[-i]] = BEHAVIOR_NOTHING
                bee.event_caring.append(CaringEvent(id, bee.care_frames))
                bee.care_frames = 0
                bee.noncare_frames = 0
                bee.frame_cared.clear()
                bee.care_hives.clear()
        else:
            bee.care_frames = 0
            bee.noncare_frames = 0
            bee.frame_cared.clear()
            bee.care_hives.clear()

    return d_caring , id


def kpdetect(filename, hivename, model, n_tracks, n_frames, n_bodyparts=3, th=0.75, draw_trajectory=False, dur=32, radian=1.5, eps=0.02):
    if not os.path.exists(f"output/{filename}/"):
        os.makedirs(f"output/{filename}/")    
    def _save(hive, img_hive_sam, c, bees: list[Bee], colors, img_tracklets):
        for bee in bees.values():
            np.savetxt(f'bees/bee_{bee.id}.txt', bee.statuses, fmt='%d')
        #print(np.sum(Bee.exchanged_series))
        #print(np.sum(Bee.hived_series))
        gen_hive_heatmap(hive, img_hive_sam, f"output/{filename}/")
        gen_graphs(c, bees, colors, f"output/{filename}/", 100)
        save_result(f"output/{filename}/", bees)
        
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
    bees: dict[int, Bee] = dict()

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

    nnds = []
    scaling_factor = np.array([1.0 / width, 1.0 / height])

    c = 0
    prog = tqdm(desc="Generating", total=n_frames)
    while True:
        success, frame = cap.read()
        if c > n_frames:
            
            fig, ax = plt.subplots()
            ax.plot([i for i in range(len(nnds))], nnds)
            fig.suptitle("密集度の時系列変化")
            ax.set_title(f"平均密集度: {np.mean(nnds)}")
            plt.savefig(f"{filename}_.png")
            _save(hive, img_hive_sam, c, bees, colors, img_tracklets)
            print(np.mean(nnds))
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
                    #cv2.circle(frame, (int(individual[i]), int(individual[i + 1])), 5, (0, 255, 0), 5)
                    cv2.circle(frame, (int(individual[i]), int(individual[i + 1])), 1, (0, 255, 0), 1)
                kpt = np.reshape(individual[:6], (3,2))
                pos_center_2 = np.mean(kpt[~np.isnan(kpt).any(axis=1), :], axis=0)
                poses = np.append(poses, [pos_center_2], axis=0)
            #X = StandardScaler().fit_transform(poses)
            X = poses * scaling_factor
            #db = DBSCAN(eps=0.3, min_samples=3)
            db = DBSCAN(eps=0.04, min_samples=3)
            db.fit(X)
            labels = db.labels_
            sum_densed[c] = len(labels[labels != -1])
            tree_total = KDTree(poses)
            D_max = np.sqrt(height**2 + width**2)
            distances_total, _ = tree_total.query(poses, k=2)

            nnd_total = np.mean(distances_total[:, 1])
            nnd_norm = nnd_total / D_max
            cohesion_index = 1.0 - nnd_norm
            cohesion_index = np.power(cohesion_index, 2)
            relative_density = sum_densed[c] / len(poses)
            CFI_double_prime = cohesion_index * relative_density

            nnds.append(CFI_double_prime)
            """
            for i, p in enumerate(poses):
                if db.labels_[i] != -1:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 17, (255, 255, 255), -1)       
                    cv2.circle(frame, (int(p[0]), int(p[1])), 15, (255, 50, 100), -1)       
            """
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
                    bees[d_int[-1]] = Bee(d_int[-1], d_int[:6], mask, pos_center, n_frames, length_trajectory=200)
                    #bees[d_int[-1]] = Bee(id=d_int[-1], pos=pos_center, length=n_frames - 1)
                    bees[d_int[-1]].feeding_hives = {h: 0 for h in hive.hives.keys()}
                else:
                    r = d_int[-1] in respowns
                    bees[d_int[-1]].update(d_int[:6], mask, pos_center, fps=fps, reset=r)
                    if draw_trajectory:
                        bees[d_int[-1]].draw_trajectory(frame, img_tracklets, colors[d_int[-1]])
                    bees[d_int[-1]].tracked_frames += 1
                    
            d_exchanges = detect_trophallaxis(bees, trackers, c, scaling_factor, fps=dur, radian=radian, eps=eps, frame=frame)
                    
            for i, bee in enumerate(bees.values()):
                if bee.id not in trackers[:, -1]:
                    bee.statuses[c] = -1
                    continue
                d_exchange = d_exchanges[bee.id]
                               
                for j in range(0, 5, 2):
                    cv2.circle(frame, (bee.kpts[j], bee.kpts[j + 1]), 2, colors[bee.id], -1)
                if not d_exchange:
                    d_caring, _ = detect_caring(bee, hive, img_hive_sam, c, fps)
                else: d_caring = False

                """
                for i in range(0, len(d_int[:3 * 2 + 1]), 2):
                    if i > 4: break
                    if mask[i] != "1":
                        #cv2.circle(frame, (d_int[i], d_int[i + 1]), 4, colors[d_int[-1]], 4)
                        if i == 0:
                            pass
                            #cv2.circle(frame, (d_int[i], d_int[i + 1]), 8, colors[d_int[-1]], 8)
                """

                cv2.putText(frame, str(bee.id), (bee.kpts[0], bee.kpts[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, str(bee.id), (bee.kpts[0], bee.kpts[1]), cv2.FONT_HERSHEY_PLAIN, 3, colors[bee.id], 2, cv2.LINE_AA)

                if d_caring:
                    cv2.circle(frame, bee.kpts_center.astype(int), 8, (255, 255, 255), -1)
                    cv2.circle(frame, bee.kpts_center.astype(int), 7, (0, 0, 255), -1)
                    #cv2.putText(frame, f"  @{str(id_hive)}", (d_int[0], d_int[1]), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 5, cv2.LINE_AA)
                    #cv2.putText(frame, f"  @{str(id_hive)}", (d_int[0], d_int[1]), cv2.FONT_HERSHEY_PLAIN, 5, colors[bee.id], 4, cv2.LINE_AA)
                    Bee.hived_series[c] += 1
                if d_exchange:
                    cv2.circle(frame, bee.kpts_center.astype(int), 8, (255, 255, 255), -1)
                    cv2.circle(frame, bee.kpts_center.astype(int), 7, (0, 255, 0), -1)
                    Bee.exchanged_series[c] += 1

            if len([bees[id].distance for id in bees if id in trackers[:, -1]and bees[id].distance != 0]) != 0 and c != 0:
                Bee.distances_avg[c] = np.mean([bees[id].distance for id in bees if id in trackers[:, -1] and bees[id].distance != 0])
                Bee.distances_med[c] = np.median([bees[id].distance for id in bees if id in trackers[:, -1]and bees[id].distance != 0])
                prog.set_description(f"{np.mean(Bee.distances_avg[:c])*100:.4f} {np.mean(Bee.distances_med[:c])*100:.4f} {np.mean(sum_densed[:c]):.4f}")
            else:
                Bee.distances_avg[c] = 0
                Bee.distances_med[c] = 0
                    
            c += 1
            prog.update(1)

            """
            img_tracklets_frame = cv2.cvtColor(img_tracklets, cv2.COLOR_BGR2BGRA)
            img_tracklets_frame[np.all(img_tracklets_frame == [0, 0, 0, 255], axis=2), 3] = 0
            x1, y1, x2, y2 = 0, 0, img_tracklets_frame.shape[1], img_tracklets_frame.shape[0]
            frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - img_tracklets_frame[:, :, 3:] / 255) + \
                                img_tracklets_frame[:, :, :3] * (img_tracklets_frame[:, :, 3:] / 255)
            """  
            
            video.write(frame)
        else: 
            _save(hive, img_hive_sam, c, bees, colors, img_tracklets)
            break
        
    return bees

def align_and_evaluate(y_true_dict, y_pred_dict):
    aligned_true = []
    aligned_pred = []
    
    common_ids = sorted(list(y_true_dict.keys()))
    
    for obj_id in common_ids:
        if obj_id in y_pred_dict:
            aligned_true.append(y_true_dict[obj_id])
            aligned_pred.append(y_pred_dict[obj_id])
        else:
            print(f"Warning: ID {obj_id} が予測データに見当たりません。")

    y_true_flat = np.concatenate(aligned_true)
    y_pred_flat = np.concatenate(aligned_pred)
    
    return y_true_flat, y_pred_flat

if __name__ == "__main__":
    model = YOLO("/kpsort/runs/obb/train5/weights/best.pt")
    #19 20 22
    #kpdetect("noflora2", "0902", model, 20, 10000)
    frame = 1000
    #frame = 100
    
    """
    dur_list = [18, 32, 64, 128]
    radian_list = [0.8, 1.2, 1.4, 1.6]
    eps_list = [0.01, 0.015, 0.02, 0.025]
    
    for dur, rad, eps in itertools.product(dur_list, radian_list, eps_list):
        try:
            bees = kpdetect("resized_0430", "resized_0430", model, 22, frame, dur=dur, radian=rad, eps=eps)
            sorted_keys = sorted(list(bees.keys()))
            sorted_dict_by_key = {k: bees[k] for k in sorted_keys}
            statuses_pred = {bee.id: bee.statuses for bee in bees.values()}
            
            from pathlib import Path
            import re
            from sklearn.metrics import classification_report
            
            path_gt = Path("/kpsort/bees_gt/")
            statuses_gt = {int(re.sub(r"\D", "", f_gt.stem)): np.loadtxt(f_gt)[:frame + 1] for f_gt in path_gt.rglob("*.txt")}
            
            y_true_flat, y_pred_flat = align_and_evaluate(statuses_gt, statuses_pred)
            
            with open("/kpsort/behavior.txt", mode='a') as f:
                f.write(f"\n\tDURATION: {dur}, rad: {rad}, eps: {eps}\n\n")
                f.write(classification_report(y_true_flat, y_pred_flat))
        except:
            continue
    """
    
    dur = 32
    rad = 1.4
    eps = 0.015
    
    bees = kpdetect("resized_0430_10000", "resized_0430", model, 22, frame, dur=dur, radian=rad, eps=eps)
    sorted_keys = sorted(list(bees.keys()))
    sorted_dict_by_key = {k: bees[k] for k in sorted_keys}
    statuses_pred = {bee.id: bee.statuses for bee in bees.values()}
    
    from pathlib import Path
    import re
    from sklearn.metrics import classification_report, confusion_matrix
    
    path_gt = Path("/kpsort/bees_gt/")
    statuses_gt = {int(re.sub(r"\D", "", f_gt.stem)): np.loadtxt(f_gt)[:frame + 1] for f_gt in path_gt.rglob("*.txt")}
    
    y_true_flat, y_pred_flat = align_and_evaluate(statuses_gt, statuses_pred)
    
    print(classification_report(y_true_flat, y_pred_flat))
    
    y_true = np.array(y_true_flat)
    y_pred = np.array(y_pred_flat)

    mask = np.isin(y_true, [0, 1, 2])
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    labels = [0.0, 1.0, 2.0]
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white")

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Rest (0)", "Contact (1)", "Trophallaxis (2)"],
                yticklabels=["Rest (0)", "Contact (1)", "Trophallaxis (2)"])

    plt.title(f'Confusion Matrix\n(DUR:{dur}, RAD:{rad}, EPS:{eps})')
    plt.ylabel('Actual Label (GT)')
    plt.xlabel('Predicted Label')

    save_path = f"test/cm_dur{dur}_rad{rad}_eps{eps}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)