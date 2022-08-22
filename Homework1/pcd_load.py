from ctypes import alignment
from unittest import result
import numpy as np
from PIL import Image
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import open3d as o3d
import copy
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("Filename")
args = parser.parse_args()
pcd = o3d.io.read_point_cloud(args.Filename)
o3d.visualization.draw_geometries([pcd])
