# Homework 1 Detail

Below is the detail of Homework1. There are three tasks in Homwork1.  

## Task1

---

1. Run load.py to get the RGB, Depth, Semantic images and camera positions.
2. Save all images in corresponding file.
3. Save all positions in GT_Pose.txt.

## Task2

---
Two part, one is convert Depth image to the point cloud, another is use the icp algorithm to align the point cloud

### Depth image to point cloud

1. First, load the RGB image and Depth image.
2. Compute intrinsic matrix from FOV (Field Of View), using 2 formula below.  
![fov1.png](https://i.stack.imgur.com/DG6tx.png)
![fov2.png](https://i.stack.imgur.com/urglA.png)

