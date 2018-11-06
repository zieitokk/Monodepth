import os
import re
import sys


## Change name of left image into name of right image

# for filename in os.listdir('/media/xiangtao/data/KITTI/data_scene_flow/training/dummy_training/left'):
#     result = re.sub(r"(\d+)\_(\d+)\.(\w+)", r"\1_11.\3", filename)





## store Images name
# img_dir = "/media/xiangtao/data/KITTI/data_scene_flow/training/dummy_training/left"
#
# left_write_file_name = "/media/xiangtao/data/KITTI/data_scene_flow/training/dummy_training/left_list.txt"
# right_write_file_name = "/media/xiangtao/data/KITTI/data_scene_flow/training/dummy_training/right_list.txt"
#
# write_left = open(left_write_file_name, 'w')
# write_right = open(right_write_file_name, 'w')
#
# for file in os.listdir(img_dir):
#     result = re.sub(r"(\d+)\_(\d+)\.(\w+)", r"\1_11.\3", file)
#     right_name = result + '\n'
#     left_name = file + '\n'
#     write_left.write(left_name)
#     write_right.write(right_name)
#
# write_left.close()
# write_right.close()
# print("done")

store = []
left_write_file_name = "/media/xiangtao/data/KITTI/data_scene_flow/training/dummy_training/left_list.txt"
right_write_file_name = "/media/xiangtao/data/KITTI/data_scene_flow/training/dummy_training/right_list.txt"
f = open(left_write_file_name, 'r')
for filename in f:
    ls = filename.split()
    for i in ls:
        store.append(i)
f.close()

