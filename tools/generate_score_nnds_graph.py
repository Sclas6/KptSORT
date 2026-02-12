import matplotlib.pyplot as plt
import pandas as pd
import japanize_matplotlib # 事前に pip install japanize-matplotlib が必要

# 1. データの準備
data = {
    "Name": ["0728_5SP_39_1", "0728_PBS_23_1", "0623_5SP_18_1", "0623_5SP_19_2", "0623_PBS_20_1"],
    "NNDS": [0.5370537881221957, 0.30318798662820234, 0.04160547680573558, 0.11300899382963742, 0.22744423908265798],
    "MISSES": [196554, 16840, 10981, 20670, 22867],
    "IDSWS": [3141, 1124, 538, 746, 530],
    "MOTA": [0.4880127371878197, 0.9219034618277303, 0.9360119543601195, 0.887295480978218, 0.883026697330267],
    "x": [0.6450, 0.9269, 0.9391, 0.8913, 0.8858]
}

df = pd.DataFrame(data).sort_values("NNDS")

# 2. グラフの描画
fig, ax1 = plt.subplots(figsize=(10, 7))

# --- 左軸: 比率・スコア系 ---
line1, = ax1.plot(df["NNDS"], df["MOTA"], marker='o', label="MOTA", color="royalblue", linewidth=2)
line2, = ax1.plot(df["NNDS"], df["x"], marker='s', label="平均生存期間", color="darkcyan", linestyle='--')
ax1.set_xlabel("密集度")
ax1.set_ylabel("比率 / スコア (MOTA, 平均生存期間)")
ax1.set_ylim(0, 1.2) # ラベル表示スペースのために少し上に余裕を持たせる
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

# --- 右軸: カウント系 (対数スケール) ---
ax2 = ax1.twinx()
line3, = ax2.plot(df["NNDS"], df["MISSES"], marker='^', label="MISSES", color="crimson", linewidth=2)
line4, = ax2.plot(df["NNDS"], df["IDSWS"], marker='d', label="IDSWS", color="forestgreen", linewidth=2)
ax2.set_ylabel("回数 (MISSES, IDSWS) ※対数スケール")
ax2.set_yscale('log')
ax2.set_ylim(10**2, 10**6) # MISSESと被らないよう表示範囲を調整

# タイトル
plt.title("密集度の変化に伴う各指標への影響", fontsize=14, pad=45)

# 凡例を結合して上部に配置
lines = [line1, line2, line3, line4]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=True)

# 3. データ名のラベル付け（重ならないように調整）
# 各ポイントごとに y軸方向にずらす値を設定
# ["0728_5SP_39_1", "0728_PBS_23_1", "0623_5SP_18_1", "0623_5SP_19_2", "0623_PBS_20_1"],
offsets = {
    "0728_5SP_39_1": (-15, 15),
    "0728_PBS_23_1": (0, 15),
    "0623_5SP_18_1": (15, 15),   # 少し左に寄せる
    "0623_5SP_19_2": (10, -20),   # 重なり回避のため下側に配置
    "0623_PBS_20_1": (0, 15)
}

for i, row in df.iterrows():
    name = row["Name"]
    off_x, off_y = offsets.get(name, (0, 10))
    
    ax1.annotate(
        name, 
        (row["NNDS"], row["MOTA"]), 
        xytext=(off_x, off_y), 
        textcoords='offset points', 
        fontsize=9,
        fontweight='bold',
        ha='center',
        # 背景を白くして、線と被っても文字を読みやすくする
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none')
    )

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig("score_nnds.png")