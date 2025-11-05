from filterpy.kalman import KalmanFilter
import pickle
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np

with open("output/flora2/trajectories_med_series.pkl", mode="rb") as f:
    series_1 = pickle.load(f)

with open("output/noflora2/trajectories_med_series.pkl", mode="rb") as f:
    series_2 = pickle.load(f)

series_1[0] = series_1[0][100:]
series_1[1] = series_1[1][100:]
series_2[0] = series_2[0][100:]
series_2[1] = series_2[1][100:]

def  LPF_KF1(x, Q):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.F = np.eye(1)
    kf.H = np.eye(1)
    kf.Q *= Q
    kf.x[0] = x[0]
    (mu, _, _, _) = kf.batch_filter(x)
    x_KF = mu[:, 0]
    return x_KF

Q = 10**(-2)
x_KF1_1 = LPF_KF1(series_1[1], 10**(-4))
x_KF2_1 = LPF_KF1(series_1[1], 10**(-5.5))
x_KF1_2 = LPF_KF1(series_2[1], 10**(-4))
x_KF2_2 = LPF_KF1(series_2[1], 10**(-5.5))

plt.figure(figsize=(80, 30))
plt.plot(series_1[0], series_1[1])
plt.plot(series_1[0], x_KF1_1, lw=4)
plt.plot(series_1[0], x_KF2_1, lw=4, color="red")
plt.suptitle("総移動距離の中央値推移", fontsize=100)
plt.title(f"平均: {np.mean(series_1[1])}cm", fontsize=80)
plt.xlabel("フレーム", fontsize=80)
plt.ylabel("移動距離の中央値", fontsize=80)
plt.savefig(f"1_med_flora.png")
plt.cla()

plt.figure(figsize=(80, 30))
plt.plot(series_2[0], series_2[1])
plt.plot(series_2[0], x_KF1_2, lw=4)
plt.plot(series_2[0], x_KF2_2, lw=4, color="red")
plt.suptitle("総移動距離の中央値推移", fontsize=100)
plt.title(f"平均: {np.mean(series_2[1])}cm", fontsize=80)
plt.xlabel("フレーム", fontsize=80)
plt.ylabel("移動距離の中央値", fontsize=80)
plt.savefig(f"1_med_noflora.png")
plt.cla()


plt.suptitle("")
plt.figure(figsize=(8, 6))
plt.boxplot([series_1[1], series_2[1], x_KF1_1.flatten(), x_KF1_2.flatten()], showmeans=True)
plt.xticks([1, 2, 3, 4], ["細菌叢あり_1", "細菌叢なし_1", "細菌叢あり_1_kf", "細菌叢なし_1_kf"])
plt.locator_params(axis='y',nbins=30)
plt.grid(axis='y', c='blue', ls=':', lw=0.3)
plt.tick_params(width = 0.01)
plt.savefig(f"boxplot2.png")
