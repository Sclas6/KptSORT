import cv2
import copy
import math
import numpy as np
import os
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

MODE_CANNY = 0
MODE_DOG = 1

BEHAVIOR_NOTHING = 0
BEHAVIOR_CARING = 1
BEHAVIOR_TROPHALLAXIS = 2

@dataclass
class CaringEvent:
    id_hive:    int
    duration:   int
    
@dataclass
class TrophallaxisEvent:
    type:       int
    id_pair:    int
    duration:   int

class Bee():
    hived_series: np.ndarray
    exchanged_series: np.ndarray
    distances_avg: np.ndarray
    distances_med: np.ndarray
    def __init__(self, id: int, kpts: np.ndarray, mask: str, pos: tuple, frames, length_trajectory: int=10):
        self.id = id
        self.age = 0
        self.distance = 0
        self.distance_sum = 0
        self.length = length_trajectory
        self.pos = pos
        self.kpts = kpts
        self.kpts_center = pos
        self.mask = mask
        #self.trajectory = pl.DataFrame({"y": [pos[0]], "x": [pos[1]]})
        self.trajectory_deque = deque([np.array(pos)], maxlen=length_trajectory)

        self.tracked_frames = 0
        self.feeding_hives = dict()
        self.exchanging = dict()
        
        self.care_frames = 0
        self.noncare_frames = 0
        self.care_hives = list()
        self.event_caring = list()
        
        self.trophallaxis_pairs = dict()
        self.nontrophallaxis_pairs = dict()
        self.pair_prevs = dict()
        self.event_trophallaxis = list()
        self.statuses = np.zeros(frames + 1)
        self.status = BEHAVIOR_NOTHING
    
    def update_status(self, status, frame):
        self.status = status
        self.statuses[frame] = status
    
    def update(self, kpts, mask, pos, fps, reset=False):
        self.kpts = kpts
        self.mask = mask
        self.kpts_center = pos
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


def DoG(img, size1, size2):
    g1 = cv2.GaussianBlur(img, (size1, size1), 0)
    g2 = cv2.GaussianBlur(img, (size2, size2), 0)
    return g1 - g2

def resize(image, size):
    h, w = image.shape[:2]
    aspect = w / h
    nh = nw = size
    if 1 >= aspect:
        nw = round(nh * aspect)
    else:
        nh = round(nw / aspect)
    resized = cv2.resize(image, dsize=(nw, nh))
    return resized

def shiroume(image, size):
    resized = resize(image, size)
    h, w = resized.shape[:2]
    x = y = 0
    if h < w:
        y = (size - h) // 2
    else:
        x = (size - w) // 2
    resized = Image.fromarray(resized)
    canvas = Image.new(resized.mode, (size, size), 255)
    canvas.paste(resized, (x, y))
    dst = np.array(canvas)
    return dst

def gen_random_colors(length: int):
    colors = set()
    while len(colors) != length:
        colors.add(tuple(np.random.randint(1, 256, size=(3,))))
    return list(colors)

class Hive:
    def __init__(self, id: int, color: tuple, pos: tuple, mask):
        self.id = id
        self.color = color
        self.pos = pos
        self.mask = mask
        self.counter = 0

class AssignBeeHive():
    def __init__(self, path_img: str, pps:int=64, cnl:int = 3, mode_binarize:int=MODE_DOG, th_size: tuple=(150, 700)):
        self.path_img = path_img
        self.dir = f"result/pps{pps}_cnl{cnl}_{mode_binarize}/"
        self.mode_binarize = mode_binarize
        self.config_sam = (pps, cnl)
        self.hives = dict()
        self.th_size = th_size
        self.colors2id = dict()
        self.center2id = dict()
        self.positions = list()
        
    def gen_binarized_image(self):
        mode = self.mode_binarize
        img = cv2.imread(self.path_img, cv2.IMREAD_GRAYSCALE)
        img = shiroume(img, max(img.shape[0], img.shape[1]))
        if mode == 0:
            blurred = cv2.medianBlur(img, ksize=5)
            blurred = cv2.equalizeHist(blurred)
            canny = cv2.Canny(img, threshold1=100, threshold2=100)
            kernel = np.ones((2,2), np.uint8)
            dilation = cv2.dilate(canny, kernel, iterations = 3)
            fixed = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
            fixed = canny
        elif mode == 1:
            fixed = DoG(img, 17, 9)
        cv2.imwrite(f"out_fixed_{mode}.png", fixed)
    
    def _optimize_mask(self, pps: int, cnl: int):
        name_img = Path(self.path_img).stem
        path = f"result/pps{pps}_cnl{cnl}_{self.mode_binarize}/"
        img =  cv2.imread(f"{path}result_pps{pps}_cnl{cnl}_{self.mode_binarize}_{name_img}_.png")
        # BGR
        """for hive in self.hives:
            color  = np.array(hive.color)
            pixels = np.count_nonzero(np.all(img==color, axis=2))
            print(pixels)"""
        # HSV
        to_del = list()
        for hive in self.hives.values():
            color = hive.color
            color_hsv = cv2.cvtColor(np.array([[color]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
            img_mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), color_hsv, color_hsv)
            pixels = np.count_nonzero(img_mask)
            if pixels < self.th_size[0]:
                to_del.append(hive.id)
            """if np.count_nonzero(img[hive.pos[1], hive.pos[0]]) == 0:
                to_del.append(hive.id)"""
        for d in reversed(to_del):
            del self.hives[d]
            
    def gen_mask_w_sam(self):
        pps, cnl = self.config_sam
        img = cv2.imread(f"out_fixed_{self.mode_binarize}.png")
        name_img = Path(self.path_img).stem
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir)
        if not os.path.exists(f"sources/hives/{name_img}/"):
            os.makedirs(f"sources/hives/{name_img}/")

        bar_sam = tqdm(total=1, desc="%-15s" % ("Preparing Mask Data"))
        if not os.path.exists(f"{self.dir}result_pps{pps}_cnl{cnl}_{self.mode_binarize}_{name_img}.pickle"):
            sam = sam_model_registry["default"](checkpoint="sources/Models/sam_vit_h_4b8939.pth")
            sam.to(device="cuda")
            mask_generator = SamAutomaticMaskGenerator(model=sam, points_per_side=pps, crop_n_layers=cnl)
            masks = mask_generator.generate(img)
            with open(f"{self.dir}result_pps{pps}_cnl{cnl}_{self.mode_binarize}_{name_img}.pickle", 'wb') as fo:
                pickle.dump(masks, fo)
        else:
            with open(f"{self.dir}result_pps{pps}_cnl{cnl}_{self.mode_binarize}_{name_img}.pickle", 'rb') as fi:
                masks = pickle.load(fi)
        bar_sam.update(1)
        bar_sam.close()

        combined_mask = np.zeros_like(img)
        np.random.seed(seed=32)
        print(len(masks))
        color_map = iter(gen_random_colors(len(masks)))
        i = 0

        for mask_data in tqdm(masks, desc="%-15s" % ("Generating Images")):
            if mask_data["area"] < self.th_size[0] or mask_data["area"] > self.th_size[1]: continue
            mask = mask_data['segmentation']
            mask = mask.astype(np.uint8)
            x, y, w, h = mask_data["bbox"]
            if w / h < 0.5 or w / h > 1.5: continue
            center = (int((x * 2 + w) / 2), int((y * 2 + h) / 2))
            color = next(color_map)
            self.hives[i] = Hive(id=i, color=color, pos=center, mask=mask)
            i += 1
            colored_mask = np.zeros_like(img)
            colored_mask[mask == 1] = color
            combined_mask[mask == 1] = colored_mask[mask == 1]
            combined_mask_colored = combined_mask.copy()
            combined_mask_colored[colored_mask > 0] = 0        
        combined_mask_3ch = np.clip(combined_mask, 0, 255)
        cv2.imwrite(f"{self.dir}result_pps{pps}_cnl{cnl}_{self.mode_binarize}_{name_img}_.png", combined_mask_3ch)
    
        self._optimize_mask(pps, cnl)
        combined_mask = np.zeros_like(img)

        for hive in tqdm(self.hives.values(), desc="%-15s" % ("Optimizing Images")):
            mask = hive.mask
            color = hive.color
            center = hive.pos
            self.colors2id[color] = hive.id
            self.center2id[center] = hive.id
            colored_mask = np.zeros_like(img)
            colored_mask[mask == 1] = color
            combined_mask[mask == 1] = colored_mask[mask == 1]
            combined_mask_colored = combined_mask.copy()
            combined_mask_colored[colored_mask > 0] = 0
            self.positions.append((hive.pos))
        combined_mask_3ch = np.clip(combined_mask, 0, 255)
        img_w_ids = copy.deepcopy(combined_mask_3ch)
        for hive in self.hives.values():
            cv2.putText(img_w_ids, str(hive.id), hive.pos, 1, 0.8, (255, 255, 255))
        img_row = cv2.imread(self.path_img)
        h, w, _ = img_row.shape
        cv2.imwrite(f"sources/hives/{name_img}/result_{name_img}.png", combined_mask_3ch[int((w-h)/2):int((w-h)/2) + h, :])
        cv2.imwrite(f"{self.dir}bbox_pps{pps}_cnl{cnl}_{self.mode_binarize}_{name_img}.png", img_w_ids)

    def gen_mask_w_sam2(self):
        pass
    
    def gen_mask_w_samHQ(self):
        pass
    
    def pos2id(self, pos: tuple, img=None):
        name_img = Path(self.path_img).stem
        if img is None:
            generated_img = cv2.imread(f"/kpsort/sources/hives/{name_img}/result_{name_img}.png")
        else:
            generated_img = img
            
        x, y = pos
        img_h = generated_img.shape[0]
        img_w = generated_img.shape[1]
        if y < img_h and y >= 0 and x < img_w and x >= 0:
            color = generated_img[(y, x)]
            if np.count_nonzero(color) != 0:
                #print("on")
                return self.colors2id[tuple(color.tolist())], 0
        x0 = np.array([xy[0] for xy in self.positions])
        y0 = np.array([xy[1] for xy in self.positions])
        distance = (x0 - x) ** 2 + (y0 - y) ** 2
        i = np.argmin(distance)
        return self.center2id[(x0[i], y0[i])], np.min(distance)