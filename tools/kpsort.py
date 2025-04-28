from __future__ import print_function
import random
import lap
import math
import numpy as np
from pykalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from tools.calk_oks import oks

np.random.seed(0)

SIGMA = 0.1
NUM_KPTS = 3

def linear_assignment(cost_matrix):
    try:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0])
    except ImportError:
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

 
def oks_batch(indivisuals_test, indivisuals_gt):
    o = np.zeros((len(indivisuals_test), len(indivisuals_gt)))
    for i, kpts_test in enumerate(indivisuals_test):
        tmp = []
        for kpts_gt in indivisuals_gt:
            tmp.append(oks(kpts_gt, kpts_test, SIGMA))
        o[i] = np.array(tmp)
    #print(o)
    return o


def fill_nan(kpts):
    for i in range(0, len(kpts) - 1, 2):
        if not math.isnan(kpts[i]):
            x = kpts[i]
            y = kpts[i + 1]
            break
    for i in range(len(kpts) - 1):
        if math.isnan(kpts[i]):
            if i % 2 == 0: kpts[i] = x
            else: kpts[i] = y
    return kpts


class KalmanKptTracker(object):
    count = 0
    def __init__(self, kpt):
        F = np.zeros((NUM_KPTS * 2 * 2 + 1, NUM_KPTS * 2 * 2 + 1))
        for i in range(NUM_KPTS * 2 * 2 + 1):
            F[i, i] = 1
            if i < NUM_KPTS * 2:
                F[i, i + NUM_KPTS * 2 + 1] = 1
        H = np.zeros((NUM_KPTS * 2 + 1, NUM_KPTS * 2 * 2 + 1))
        for i in range(NUM_KPTS * 2 + 1):
            H[i, i] = 1
        R = np.zeros((NUM_KPTS * 2 + 1, NUM_KPTS * 2 + 1))
        for i in range(NUM_KPTS * 2 + 1):
            if i != NUM_KPTS * 2:
                R[i, i] = 10
        Q = np.zeros((NUM_KPTS * 2 * 2 + 1, NUM_KPTS * 2 * 2 + 1))
        for i in range(NUM_KPTS * 2 * 2 + 1):
            Q[i, i] = 100
        Q[NUM_KPTS * 2 + 1:, NUM_KPTS * 2 + 1:] *= 0.01
        P = np.zeros((NUM_KPTS * 2 * 2 + 1, NUM_KPTS * 2 * 2 + 1))
        for i in range(NUM_KPTS * 2 * 2 + 1):
            P[i,i] = 10
        P[NUM_KPTS * 2 + 1:, NUM_KPTS * 2 + 1:] *= 1000
        
        self.kf = KalmanFilter(transition_matrices=F, observation_matrices=H, transition_covariance=Q, observation_covariance=R)
        self.est_p = P
        self.est_x = np.zeros((NUM_KPTS * 2 * 2 + 1,))
        kpt = fill_nan(kpt)
        self.est_x[:NUM_KPTS * 2 + 1] = kpt
        
        self.time_since_update = 0
        self.id = KalmanKptTracker.count
        KalmanKptTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self, kpt):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.age = 0
        #obs = fill_nan(kpt)
        obs = np.ma.masked_where(np.isnan(kpt), kpt)
        mask_pre = str(bin(int(self.est_x[NUM_KPTS * 2])))[2:].zfill(NUM_KPTS * 2)
        mask_obs = str(bin(int(obs[NUM_KPTS * 2])))[2:].zfill(NUM_KPTS * 2)
        self.est_x, self.est_p = self.kf.filter_update(self.est_x, self.est_p, obs)
        self.est_x[NUM_KPTS * 2] = obs[NUM_KPTS * 2]
        if self.est_x[6] != 0:
            self.est_x[:7] = fill_nan(obs)
        """if self.est_x[6] != 0:
            for i, m in enumerate(mask_obs):
                if m != "1":
                    x = obs[i]
                    y = obs[i + 1]
                    break
            for i, m in enumerate(mask_obs):
                if m == "1":
                    if i % 2 == 0:  
                        self.est_x[i] = x
                        self.est_x[i + NUM_KPTS * 2 + 1] = 0
                        self.est_p[i, i] = 10
                    else:
                        self.est_x[i] = y
                        self.est_x[i + NUM_KPTS * 2 + 1] = 0
                        self.est_p[i, i] = 10
        for i, (b_pre, b_obs) in enumerate(zip(mask_pre, mask_obs)):
            if b_obs == "0" and b_pre == "1":
                self.est_x[i] = obs[i]
                self.est_x[i + NUM_KPTS * 2 + 1] = 0
                self.est_p[i, i] = 10"""
                
    def predict(self):
        self.est_x, self.est_p = self.kf.filter_update(self.est_x, self.est_p)
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.est_x)
        return self.history[-1]
    
    def get_state(self):
        #return self.est_x[:NUM_KPTS * 2 + 1]
        return self.est_x
    
    
def associate_detections_to_trackers(keypoints,trackers, oks_threshold = 0.8):
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(keypoints)), np.empty((0,5),dtype=int)  
    oks_matrix = oks_batch(keypoints, trackers) 
    #print(oks_matrix)
    #print(trackers)
    if min(oks_matrix.shape) > 0:
        a = (oks_matrix > oks_threshold).astype(np.int32)
        #print(a)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-oks_matrix)
            """print(matched_indices)
            matched_indices = list()
            selected_0 = set()
            selected_1 = set()
            b = np.sort(oks_matrix[oks_matrix!=0])[::-1]
            for i in b:
                pos = np.where(oks_matrix==i)
                if pos[0][0] not in selected_0:
                        matched_indices.append([pos[0][0], pos[1][0]])
                        selected_0.add(pos[0][0])
            matched_indices = np.array(sorted(matched_indices))
            print(matched_indices)"""
    else:
        matched_indices = np.empty(shape=(0,2)) 
    unmatched_detections = []
    for d, det in enumerate(keypoints):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers[:,:6]):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)
            #print(f"{t} Died")    
    matches = []
    for m in matched_indices:
        if(oks_matrix[m[0], m[1]]<oks_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
            #print(f"{m[1]} Died")
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=0, min_hits=3, oks_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.oks_threshold = oks_threshold
        self.trackers = []
        self.frame_count = 0
        self.lost_tracks = []

    def update(self, kpts=np.empty((0, 3)), desirable2removes=[], oks_threshold=0.4):
        """kpts - [[x1, y1, x2, y2, x3, y3],...]"""
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 7))
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            #pos = self.trackers[t].get_state()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
        if len(trks) != 0 and len(desirable2removes) !=0:
            #print(desirable2removes)
            removes = list()
            for pair_d2r in desirable2removes:
                okses = list()
                for d2r in pair_d2r:
                    oks_max = 0
                    for trk in trks:
                        oks_max = max(oks(kpts[d2r], trk, SIGMA), oks_max)
                    okses.append(oks_max)
                okses = np.array(okses)
                
                if np.all(okses==min(okses)):
                    removes.append(pair_d2r[0])
                else:
                    removes.append(pair_d2r[np.where(okses==min(okses))[0][0]])
            kpts = np.delete(kpts, removes, 0)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(kpts, trks, self.oks_threshold)
        
        # respawn tracker
        for i in unmatched_trks:
            #print(self.trackers[i].id)
            self.lost_tracks.append((self.trackers[i], self.frame_count))
            
		# update matched trackers with assigned detections
        for m in matched:
            #self.trackers[m[1]].update(kpts[m[0], :])
            self.trackers[m[1]].update(kpts[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            #if False:
            if len(self.lost_tracks) != 0:
                for j in range(len(self.lost_tracks)):
                    score = oks(self.lost_tracks[j][0].get_state()[:NUM_KPTS * 2 + 1], kpts[i, :7], SIGMA)
                    if score >= oks_threshold:
                        #print(score)
                        self.trackers.append(self.lost_tracks[j][0])
                        self.trackers[-1].update(kpts[i, :7])
                        #print(self.lost_tracks[j].id)
                        self.lost_tracks.pop(j)
                        #print(f"Restored: {self.lost_tracks[j][0].get_state()} -> {kpts[i, :7]}")
                        break
                    if j == len(self.lost_tracks) - 1:
                        trk = KalmanKptTracker(kpts[i,:])
                        self.trackers.append(trk)
            else:
                trk = KalmanKptTracker(kpts[i,:])
                self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if True:
                ret.append(np.concatenate((d,[trk.id])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers[i].age = 0
                self.lost_tracks.append((self.trackers.pop(i), self.frame_count))
                #self.trackers.pop(i)
        for i, trk in enumerate(self.lost_tracks):
            if self.frame_count - trk[1] > 1000:
                self.lost_tracks.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))