#from tools.AssignBeeHive import AssignBeeHive
from tools.beehive_image_generator.BIG import *
import pickle
import os

hivename = "11105SP"
path_out = f"/bee/KptSORT/sources/hives/{hivename}"

pano = cv2.imread("/bee/KptSORT/tools/beehive_image_generator/output/1110/Stitched.png")
#pano = generate_stitched_hive("1110")
hive = cv2.imread("/bee/KptSORT/tools/beehive_image_generator/sources/1110/11105SP.png")
#warped = cv2.imread("/home/lab/lab/bee/BIGv2/output/Warped.png")
#warped_tps = cv2.imread("/home/lab/lab/bee/BIGv2/output/Warped_TPS.png")
#test = cv2.imread("/home/lab/lab/bee/BIGv2/sources/0728/IMG_0019.jpg")

#generate_linearly_transformed_img(path_out, pano, hive)
generate_nonlinearly_transformed_img_with_sift(path_out, pano, hive)
exit()
name_file = "11105SP.png"
#hive = AssignBeeHive(f"sources/hives/{name_file}", th_size=(190, 1250))
hive = AssignBeeHive(f"/bee/KptSORT/sources/hives/11105SP/{name_file}", th_size=(150, 700))
#hive = AssignBeeHive("sources/hives/hive_yellow.png")
#hive = AssignBeeHive("tools/out.png")
hive.gen_binarized_image()
hive.gen_mask_w_sam()
with open(f"/bee/KptSORT/sources/hives/11105SP/{name_file[:-4]}.pickle", "wb") as f:
    pickle.dump(hive, f)