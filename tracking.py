import os
os.chdir("/kpsort")
from tools.kpsort import Sort
from tools.loadpkl_jit import *
from ultralytics import YOLO
from tqdm import tqdm
import cv2
from collections import deque
import numpy as np
import pickle
import matplotlib.pyplot as plt
import japanize_matplotlib
from concurrent.futures import ProcessPoolExecutor
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
import shutil
from PIL import Image

MODE_GT = 0
MODE_AUTO = 1

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

def kpdetect(filename, model, n_tracks, n_frames, n_bodyparts=3, th=0.75, mode=MODE_AUTO):
    def _save(mot_results, trackers_result, nnds):
        mot_output_path = f"output/{filename}/{filename}.txt"
        kptsort_output_path = f"output/{filename}/trackers.npz"
        lines = []
        for row in mot_results:
            line = ",".join([
                str(int(row[0])),    # Frame
                str(int(row[1])),    # ID
                f"{row[2]:.2f}",     # Left
                f"{row[3]:.2f}",     # Top
                f"{row[4]:.2f}",     # Width
                f"{row[5]:.2f}",     # Height
                "1", "-1", "-1", "-1"
            ])
            lines.append(line)
        with open(mot_output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        #with open(kptsort_output_path, mode="wb") as f:
        #    pickle.dump(trackers_result, f)
        np.savez_compressed(kptsort_output_path, *trackers_result)
        print(f"MOT results saved to: {mot_output_path} and {kptsort_output_path}")
        fig, ax = plt.subplots()
        ax.plot([i for i in range(len(nnds))], nnds)
        fig.suptitle("密集度の時系列変化")
        ax.set_title(f"平均密集度: {np.mean(nnds)}")
        plt.savefig(f"output/{filename}/nnd_mean.png")
        print("\tNNDS: ", np.mean(nnds))
        
    if not os.path.exists(f"output/{filename}/"):
        os.makedirs(f"output/{filename}/")    

    path_csv = f"sources/{filename}/CTD.csv"
    path_pkl = f"sources/{filename}/BU.pickle"

    with open(path_pkl, "rb") as file:
        data_pkl: dict = pickle.load(file)
    data_csv = load_csv(path_csv, n_tracks, n_bodyparts)
    color_map = iter(gen_random_colors(10000, 334))

    cap = cv2.VideoCapture(f"sources/{filename}/{filename}.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scaling_factor = np.array([1.0 / width, 1.0 / height])
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(frames, n_frames)
    print(f"Video\t: {filename}.mp4\nFrames\t: {n_frames}\n FPS\t: {fps}")

    colors = dict()
    sum_densed = np.zeros((n_frames + 1))
    mot_tracker = Sort(oks_threshold=0.00001, individuals=n_tracks, max_age=int(fps))
    mot_results = []
    kpsort_results = []
    nnds = []
    
    frame_buffer = deque(maxlen=int(fps * 1.5))
    resize_rate = 0.5
    debug_dir = "debug_frames"
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir, exist_ok=True)

    data_raw = list()
    for i, _ in tqdm(enumerate(data_pkl), total=len(data_pkl.keys())):
        if i > 0:
            data_raw.append(pkl2setlist(data_pkl, i - 1))


    c = 0
    prog = tqdm(desc="Generating", total=n_frames)
    while True:
        success, frame = cap.read()
        if c > n_frames:
            _save(mot_results, kpsort_results, nnds)
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
            
            trackers, respowns = mot_tracker.update(individuals, desirable2remove, th)

            if mode == MODE_GT:
            # --- フレームバッファの更新 (色の描画を共通化) ---
                annotated_frame = frame.copy()
                for d in trackers:
                    d_int = d.astype(np.int32)
                    if d_int[-1] not in colors:
                        colors[d_int[-1]] = next(color_map)
                    tid = int(d[-1])
                    color = colors[d_int[-1]]
                    
                    # 重心に点を描画
                    cv2.circle(annotated_frame, (int(d[0]), int(d[1])), 4, color, -1)
                    # IDテキストも同じ色で描画
                    cv2.putText(annotated_frame, f"ID:{tid}", (int(d[0]), int(d[1])-10), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(annotated_frame, f"ID:{tid}", (int(d[0]), int(d[1])-10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2, cv2.LINE_AA)
                    if d_int[-1] in respowns:
                        cv2.circle(annotated_frame, (int(d[0]), int(d[1])), 20, (0, 0, 255), 5)
                
                h, w = annotated_frame.shape[:2]
                small_frame = cv2.resize(annotated_frame, (int(w * resize_rate), int(h * resize_rate)))
                frame_buffer.append(small_frame)

                # --- 手動修正ブロック ---
                """
                if len(respowns) > 0 and c > 1:
                    # 1. 保存用ディレクトリの作成
                    event_dir = os.path.join(debug_dir, f"event_f{c:05d}")
                    os.makedirs(event_dir, exist_ok=True)
                    
                    # 2. バッファ内の各フレームを証拠画像として保存
                    for i, f_img in enumerate(frame_buffer):
                        f_num = c - (len(frame_buffer) - 1) + i
                        img_path = os.path.join(event_dir, f"seq_{i:02d}_f{f_num:05d}.png")
                        cv2.imwrite(img_path, f_img)
                    
                    print(f"\n[!!!] ID Respawn detected at Frame {c}.")
                    print(f"Check images in: {event_dir}") 
                """                    
                if len(respowns) > 0 and c > 1:
                    video_path = os.path.join(debug_dir, f"event_f{c:05d}.mp4")
                    if len(frame_buffer) > 0:
                        # 動画書き出し設定
                        fourcc = cv2.VideoWriter_fourcc(*'H264') # または 'avc1'
                        buf_h, buf_w = frame_buffer[0].shape[:2]
                        out = cv2.VideoWriter(video_path, fourcc, fps, (buf_w, buf_h))
                        
                        for f in frame_buffer:
                            out.write(f)
                        out.release()
                        
                        print(f"\n[!!!] ID Respawn at Frame {c}. Video saved: {video_path}")
                        print(f"Check Video at: {video_path}")
                        print(f"Current IDs in this frame: {trackers[:, -1].astype(int)}")
                        print(f"New (Respawned) IDs to check: {respowns}")

                    # 3. 修正指示の集約
                    id_map = {}
                    ids_to_delete = []
                    for rid in respowns:
                        val = input(f"Enter correct ID for ID {rid} (Delete: 'd', Keep: Enter): ")
                        if val.strip().lower() == 'd':
                            ids_to_delete.append(rid)
                        elif val.strip() != "":
                            id_map[rid] = int(val)

                    # 4. 物理削除処理 (逆順popでメモリから完全に抹消)
                    for rid in ids_to_delete:
                        # 表示用配列から削除
                        trackers = trackers[trackers[:, -1] != rid]
                        # SORT内部メモリから削除
                        for trk_idx in range(len(mot_tracker.trackers) - 1, -1, -1):
                            if mot_tracker.trackers[trk_idx][0].id == rid:
                                mot_tracker.trackers.pop(trk_idx)
                                print(f"  -> Deleted ID {rid} from memory.")

                    # 5. 安全なIDスワップロジック (三すくみ対応)
                    # 手順A: 変更対象を一時的な負のIDに退避させて衝突を防ぐ
                    temp_map = {} # {修正対象の元のID: 一時ID}
                    for i, (old_id, new_id) in enumerate(id_map.items()):
                        temp_id = -(i + 1) 
                        temp_map[old_id] = temp_id
                        for trk_idx in range(len(mot_tracker.trackers)):
                            if mot_tracker.trackers[trk_idx][0].id == old_id:
                                mot_tracker.trackers[trk_idx][0].id = temp_id
                                print(f"  -> Temp-move: {old_id} to {temp_id}")

                    # 手順B: 目的地(new_id)が既存IDなら、上書きのために古い方を消去
                    for new_id in id_map.values():
                        for trk_idx in range(len(mot_tracker.trackers) - 1, -1, -1):
                            # 退避させた一時ID（負数）以外で、new_idと被る既存個体を消す
                            if mot_tracker.trackers[trk_idx][0].id == new_id:
                                mot_tracker.trackers.pop(trk_idx)
                                print(f"  -> Cleared existing ID {new_id} to overwrite.")

                    # 手順C: 一時IDから最終的な目的地(new_id)へ割り当て
                    for old_id, new_id in id_map.items():
                        temp_id = temp_map[old_id]
                        for trk_idx in range(len(mot_tracker.trackers)):
                            if mot_tracker.trackers[trk_idx][0].id == temp_id:
                                trk_obj = mot_tracker.trackers[trk_idx][0]
                                trk_obj.id = new_id
                                # パラメータを最強に固定して「戻り」を防止
                                trk_obj.hits = 999 
                                trk_obj.hit_streak = 100
                                trk_obj.time_since_update = 0
                                # 速度ベクトル(Kalman stateの後ろ半分)をリセットして予測を安定させる
                                trk_obj.est_x[6 + 1:] = 0 # NUM_KPTS*2 + 1 以降を0に
                                print(f"  -> Final-move: {old_id} (via {temp_id}) to {new_id}")
                        
                        # 表示用配列のIDも更新
                        trackers[trackers[:, -1] == old_id, -1] = new_id

                    print(f"  -> Frame {c} manual correction finalized.\n")

            kpsort_results.append(trackers)
            for d in trackers:
                d_int = d.astype(np.int32)
                if mode == MODE_AUTO:
                    if d_int[-1] not in colors:
                        colors[d_int[-1]] = next(color_map)
                kpts_raw = d[:6]
                kpts_x = kpts_raw[0::2]
                kpts_y = kpts_raw[1::2]
                
                valid_x = kpts_x[~np.isnan(kpts_x)]
                valid_y = kpts_y[~np.isnan(kpts_y)]
                if len(valid_x) > 0:
                    bb_left = np.min(valid_x)
                    bb_top = np.min(valid_y)
                    bb_width = np.max(valid_x) - bb_left
                    bb_height = np.max(valid_y) - bb_top
                    obj_id = int(d[-1])
                    
                    mot_results.append([
                        c + 1, obj_id, bb_left, bb_top, bb_width, bb_height, 1, -1, -1, -1
                    ])

            c += 1
            prog.update(1)
            
        else: 
            _save(mot_results, kpsort_results, nnds)
            break

if __name__ == "__main__":
    
    model = YOLO("/kpsort/runs/obb/train5/weights/best.pt")
    kpdetect("flora1", model, 18, 1000, mode=MODE_GT)
