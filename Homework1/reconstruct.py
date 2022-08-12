import numpy as np
from PIL import Image
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2


def depth_image_to_point_cloud():
    pass

# read file
f= open("Pose_data_1F.txt", 'r')

line= f.readline()
Position_list=line.split()
print(type(Position_list[0]))

for idx, item in enumerate(Position_list):
    item = float(item)
    Position_list[idx]=item


print(np.array(Position_list))






f.close()

