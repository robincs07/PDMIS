import numpy as np
import matplotlib.pyplot as plt
import cv2

floor_color = np.asarray([255.0, 194.0, 7.0])

def creat_map():
    """Create a map from the given file.
    """
    points = np.load("../semantic_3d_pointcloud/point.npy")
    colors = np.load("../semantic_3d_pointcloud/color0255.npy")

    points[:, [0, 2]]*= -1
    # rescale points
    points = points*10000./255.

    avg_height = np.average(points[:, 1])
    print(avg_height)
    cond_points = np.logical_and(points[:, 1] > avg_height-1 , points[:, 1] < avg_height-0.2)

    index = np.where(np.all(colors == floor_color, axis=1)) # find the index of the floor
    floor_p = points[index]
    floor_c = colors[index]
    colors = colors[cond_points]
    points = points[cond_points]
    print("num of points: {}".format(points.shape))


    with open("points.txt", "w") as f:
        for point in points:
            f.write(f"{point[0]},{point[1]},{point[2]}\n")

    with open("colors.txt", "w") as f:  
        for color in colors:
            f.write(f"{color[0]},{color[1]},{color[2]}\n")



    # create map from points
    fig = plt.figure()
    plt.scatter(points[:, 2], points[:, 0], s=1, c=colors/255.)
    plt.axis('off')
    plt.savefig("map.png")
    plt.show()
    fig = plt.figure()

    plt.scatter(floor_p[:, 2], floor_p[:, 0], s=5, c=(floor_c)/255.)
    plt.axis('off')
    plt.savefig("floor.png")
    plt.show()


if __name__ == "__main__":
    creat_map()
