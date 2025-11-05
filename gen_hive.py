from tools.AssignBeeHive import AssignBeeHive
import pickle

name_file = "0728.png"
#hive = AssignBeeHive(f"sources/hives/{name_file}", th_size=(190, 1250))
hive = AssignBeeHive(f"sources/hives/{name_file}", th_size=(150, 700))
#hive = AssignBeeHive("sources/hives/hive_yellow.png")
#hive = AssignBeeHive("tools/out.png")
hive.gen_binarized_image()
hive.gen_mask_w_sam()
with open(f"sources/hives/0728/{name_file[:-4]}.pickle", "wb") as f:
    pickle.dump(hive, f)