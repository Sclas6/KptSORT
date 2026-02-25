from __future__ import print_function
import random
import lap
import math
import numpy as np
from pykalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from tools.calk_oks import oks
from numba import njit
import copy

np.random.seed(0)

SIGMA = 1.0
NUM_KPTS = 3

def linear_assignment(cost_matrix):
    try:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0])
    except ImportError:
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

@njit
def oks_batch(indivisuals_test, indivisuals_gt):
    o = np.zeros((len(indivisuals_test), len(indivisuals_gt)))
    for i, kpts_test in enumerate(indivisuals_test):
        tmp = []
        for kpts_gt in indivisuals_gt:
            tmp.append(oks(kpts_gt, kpts_test, SIGMA))
        o[i] = np.array(tmp)
    #print(o)
    return o

def fill_masked(kpts):
    for i in range(0, len(kpts) - 1, 2):
        if not np.ma.is_masked(kpts[i]):
            x = kpts[i]
            y = kpts[i + 1]
            break
    for i in range(len(kpts) - 1):
        if np.ma.is_masked(kpts[i]):
            if i % 2 == 0: kpts[i] = x
            else: kpts[i] = y
    return kpts

@njit
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
    def __init__(self, kpt, id):
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
        self.id = id
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
            self.est_x[:7] = fill_masked(obs)
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
        #print(oks_matrix)
            
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
    def __init__(self, max_age=3, min_hits=3, oks_threshold=0.3, individuals=0):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.oks_threshold = oks_threshold
        self.individuals = individuals
        self.trackers = []
        self.frame_count = 0
        #self.lost_tracks = []
        
    def search(self, id):
        for t in self.trackers:
            if t.id == id:
                return t

    def update(self, kpts=np.empty((0, 3)), desirable2removes=[], oks_threshold=0.4):
        """kpts - [[x1, y1, x2, y2, x3, y3],...]"""
        self.frame_count += 1
        # get predicted locations from existing trackers.
        #trks = np.zeros((len([trk for trk in self.trackers if trk[1] == -1]), 7))
        trks = np.zeros((len(self.trackers), 7))
        ret = []
        for t, trk in enumerate(trks):
            #pos = [trk for trk in self.trackers if trk[1] == -1][t][0].predict()
            pos = self.trackers[t][0].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
        if len(desirable2removes) !=0:
            removes = list()
            if len(trks) != 0:
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
            else:
                for pair_d2r in desirable2removes:
                    removes.append(np.random.choice(pair_d2r, 1)[0])
            kpts = np.delete(kpts, removes, 0)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(kpts, trks, self.oks_threshold)
        #print(matched)
            
		# update matched trackers with assigned detections
        matched_rm = []
        ids = set()
        for i, m in reversed(list(enumerate(matched))):
            det_pos = np.array([np.average(kpts[m[0]][:6:2][~np.isnan(kpts[m[0]][:6:2])]), np.average(kpts[m[0]][1:6:2][~np.isnan(kpts[m[0]][1:6:2])])])
            trk_pos = np.array([np.average(self.trackers[m[1]][0].get_state()[:6:2][~np.isnan(self.trackers[m[1]][0].get_state()[:6:2])]), np.average(self.trackers[m[1]][0].get_state()[1:6:2][~np.isnan(self.trackers[m[1]][0].get_state()[1:6:2])])])
            dist = np.linalg.norm(det_pos - trk_pos)
            #if True:
            if dist < 50:
                self.trackers[m[1]][0].update(kpts[m[0], :])
                self.trackers[m[1]][1] = -1
            else:
                unmatched_dets = np.append(unmatched_dets, int(m[0])).astype(int)
                unmatched_trks = np.append(unmatched_trks, int(m[1])).astype(int)
        ids = {t[0].id for t in self.trackers}
        ids_available = list(set([i for i in range(self.individuals)]) - ids)
        respowns = list()

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            if self.frame_count != 1:
                score_max = -1
                specious_score = -1
                for j, t in enumerate(self.trackers):
                    if t[1] == -1:
                        continue
                    else:
                        score = oks(t[0].get_state()[:NUM_KPTS * 2 + 1], kpts[i, :7], SIGMA)
                        score_max = max(score, score_max)
                        if score == score_max:
                            specious_score = j
                if specious_score != -1 and score_max >= oks_threshold:
                    self.trackers[specious_score][1] = -1
                    self.trackers[specious_score][0].update(kpts[i, :7])
                elif len(ids_available) != 0:
                    trk = KalmanKptTracker(kpts[i,:], ids_available.pop())
                    self.trackers.append([trk, -1])
                    respowns.append(self.trackers[-1][0].id)
                else:
                    pass
                 
            elif len(ids_available) != 0:
                trk = KalmanKptTracker(kpts[i,:], ids_available.pop())
                self.trackers.append([trk, -1])
                respowns.append(self.trackers[-1][0].id)
                
        # respawn tracker
        for i in unmatched_trks:
            if self.trackers[i][1] == -1:
                self.trackers[i][1] = self.frame_count

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk[0].get_state()
            if trk[1] == -1:
                ret.append(np.concatenate((d,[trk[0].id])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if(trk[0].time_since_update > self.max_age):
                self.trackers[i][0].age = 0
                self.trackers.pop(i)
        self.trackers = [trk for trk in self.trackers if not (trk[1] != -1 and self.frame_count - trk[1] > self.max_age)]

        if(len(ret)>0):
            return np.concatenate(ret), respowns
        return np.empty((0,5)), respowns