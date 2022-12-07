import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

color_map = {"refrigerator": [255.0, 0.0, 0.0],
            "rack": [0.0, 255.0, 133.0],
            "lamp": [160.0, 150.0, 20.0],
            "cooktop": [7.0, 255.0, 224.0],
            "cushion": [255.0, 9.0, 92.0],
            "floor": [255.0, 255.0, 255.0],
            "floor_map": [255.0, 194.0, 7.0]
}


class nodes:
    def __init__(self, x, y):
        self.point = [x, y]
        self.parent = []

def line(x1, y1, x2, y2):
    """Draw a line pixel between two points
    """
    print("x1: {}, y1: {}, x2: {}, y2: {}".format(x1, y1, x2, y2))
    line_pixel = []
    steep = abs(y2 - y1) > abs(x2 - x1)
    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    if x1>x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    deltax = x2-x1
    deltay = abs(y2-y1)
    error = deltax/2
    y= y1
    if y1<y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1+1, x2):
        if steep:
            line_pixel.append([y, x])
        else:
            line_pixel.append([x, y])
        error = error - deltay
        if error<0:
            y = y + ystep
            error = error + deltax
    # print("x1: {}, y1: {}, x2: {}, y2: {}".format(x1, y1, x2, y2))
    # print("line_pixel: {}".format(line_pixel))

    return line_pixel

def distance(x1, y1, x2, y2):
    """Compute the distance between two points
    """
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

        

def pixel_coord_np(height, width):
    """get pixel coordinates in numpy array
    """
    x = np.arange(width)
    y = np.arange(height)
    [X, Y] = np.meshgrid(x, y)
    return np.vstack((X.flatten(), Y.flatten())).T

def find_nearest_point(point, points):
    """Find the nearest point to the given point
    """
    dist = np.linalg.norm(points - point, axis=1)
    nearest_point = points[np.argmin(dist)]
    return nearest_point



def click_event(event, x, y, flags, param):
    """Get the coordinates of the clicked point
    """
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("start Point x: {}, y: {}".format(x, y))
        start_point.append(x)
        start_point.append(y)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("image", img)

def find_nearest_node(point, tree):
    """Find the nearest node in the tree
    """
    node_list = []
    for i in range(len(tree)):
        node_list.append(tree[i].point)
    node_list = np.asarray(node_list)
    point = np.asarray(point)
    dist = np.linalg.norm(node_list - point, axis=1)
    nearest_node = tree[np.argmin(dist)]
    return nearest_node

def compute_angle(point1, point2):
    """Compute the angle between two points
    """
    point1 = np.squeeze(point1)
    x1, y1 = point1
    x2, y2 = point2
    return np.arctan2(y1-y2, x1-x2)

def collision(sx, sy, rx, ry):
    """Check if there is a collision between two point
    """
    print("sx: {}, sy: {}, rx: {}, ry: {}".format(sx, sy, rx, ry))
    if sx == rx and sy == ry:
        return False, []
    if sx == rx:
        Y = np.arange(sy, ry, (ry-sy)/10.0, dtype=np.float128)
        X = ((rx-sx)/(ry-sy))*(Y-sy) + sx
    else:
        X = np.arange(sx, rx, (rx-sx)/10.0, dtype=np.float128)
        Y = ((ry-sy)/(rx-sx))*(X-sx) + sy
    line_pixel = np.vstack((X, Y)).T
    line_pixel = line(int(sx) , int(sy), int(rx), int(ry))
    # print("line_pixel: {}".format(line_pixel))
    img_ori = cv2.imread("map.png")
    for i in range(len(line_pixel)):
        if not np.all(img_ori[int(line_pixel[i][1]), int(line_pixel[i][0])] == [255.0, 255.0, 255.0]):
            return True, line_pixel
    return False, line_pixel

def check_collision(point, node, step_size):
    """Check  if there is a collision between the point and the node
    """
    point = np.asarray(point)
    node = np.asarray(node)
    theta = compute_angle(point, node)
    dist = np.linalg.norm(point - node)
    if dist < step_size:
        step_size = dist
    sx, sy = node[0]+np.cos(theta)*step_size, node[1]+np.sin(theta)*step_size
    #check collision between sampled point and node
    colli, line_pixel = collision(sx, sy, node[0], node[1])

    ####################### show the line pixel ###########################
    # img_ori = cv2.imread("map.png")
    # for i in range(len(line_pixel)):
    #     print("image line color: {}".format(img_ori[int(line_pixel[i][1]), int(line_pixel[i][0])]))
    # img3 = img.copy()
    # for i in range(len(line_pixel)):
    #     cv2.circle(img3, (int(line_pixel[i][1]), int(line_pixel[i][0])), 1, (0, 0, 0), -1)
    # cv2.imshow("image", img3)
    # cv2.imshow("image_ori", img_ori)
    # cv2.waitKey(0)
    #######################################################################
    if colli:
        node_conn = False
    else:
        node_conn = True
    
    #check collision between sampled point and end point
    colli, line_pixel = collision(sx, sy, end_point[0], end_point[1])
    if colli:
        direct_conn = False
    else:
        direct_conn = True
    
    return (int(sx), int(sy), direct_conn, node_conn)
    




def RRT(img, img2, start_point, end_point, step_size):
    """Rapidly-exploring Random Tree
    """
    tree =[]
    tree.append(nodes(start_point[0], start_point[1]))
    tree[0].parent.append(start_point)
    cv2.circle(img2, (start_point[0], start_point[1]), 3, (0, 0, 255), -1)
    cv2.circle(img2, (end_point[0], end_point[1]), 3, (0, 0, 0), -1)
    cv2.imshow("out", img2)
    h, w = img.shape[:2]
    i=1
    path_found = False
    while not path_found:
        random_point = nodes(random.randint(0, w), random.randint(0, h))
        near_node = find_nearest_node(random_point.point, tree)
        tx, ty, direct_conn, node_conn = check_collision(random_point.point, near_node.point, step_size)
        
        # check if the sample point can connect to the end point and the node
        if direct_conn and node_conn:
            print("node can direct connect to end point")
            tree.append(nodes(tx, ty))
            tree[i].parent = near_node.parent.copy()
            tree[i].parent.append(near_node.point)
            path = tree[i].parent.copy()
            path.append([int(tx), int(ty)])
            path.append([end_point[0], end_point[1]])
            path = np.asarray(path)
            np.save("path.npy", path)

            cv2.circle(img2, (int(tx), int(ty)), 3, (0, 0, 255), -1)
            cv2.line(img2, (int(tx), int(ty)), (int(near_node.point[0]), int(near_node.point[1])), (0, 0, 255), 2)
            cv2.line(img2, (int(tx), int(ty)), (int(end_point[0]), int(end_point[1])), (0, 0, 255), 2)
            cv2.imshow("out", img2)
            print("Path found")
            #line the parent node
            for j in range(len(tree[i].parent)-1):
                cv2.line(img2, (int(tree[i].parent[j][0]), int(tree[i].parent[j][1])), (int(tree[i].parent[j+1][0]), int(tree[i].parent[j+1][1])), (0, 0, 255), 2)
            cv2.imshow("out", img2)
            cv2.imwrite("RRT_{}.png".format(target), img2)
            break
        # check if the sample point can connect to the node
        elif node_conn:
            print("nodes connected")
            tree.append(nodes(tx, ty))
            tree[i].parent = near_node.parent.copy()
            tree[i].parent.append(near_node.point)
            i=i+1
            cv2.circle(img2, (int(tx), int(ty)), 2, (100, 0, 0), -1)
            cv2.line(img2, (int(tx), int(ty)), (int(near_node.point[0]), int(near_node.point[1])), (0, 0, 255), 1)
            cv2.imshow("out", img2)
            cv2.waitKey(3)
            continue
        else:
            print("No direct con. and no node con. :( Generating new rnd numbers")
            continue
    else:
        print("Path not found")




if __name__ == "__main__":
    print("Our selected target is: (refrigerator, rack, lamp, cooktop, cushion)\nInput your target:")
    target = input()
    # target = "refrigerator"

    img = cv2.imread("map.png") # read map
    img2 = img.copy()
    height, width, _ = img.shape
    points = pixel_coord_np(height, width)
    colors = np.asarray(img, dtype=np.float32).reshape(height*width, 3)
    colors = colors[:, ::-1]

    floor_img = cv2.imread("floor.png") # read floor map
    height, width, _ = img.shape
    floor_p = pixel_coord_np(height, width)
    floor_c = np.asarray(floor_img, dtype=np.float32).reshape(height*width, 3)
    floor_c = floor_c[:, ::-1]
    
    index = np.where((floor_c == color_map["floor_map"]).all(axis=1))
    floor_p = floor_p[index]

    index = np.where(np.all(colors == color_map[target], axis=1)) # find the index of the target
    refri_points = points[index]
    mean_point = np.mean(refri_points, axis=0)
    index = np.where(np.all(colors == color_map["floor"], axis=1)) # find the index of the floor
    floor = points[index]

    # define the search area
    if target == "refrigerator":
        search_space = floor[np.where((floor[:, 0]<mean_point[0])&(floor[:, 1]<mean_point[1]))]
    elif target == "cooktop":
        search_space = floor[np.where(floor[:, 1]>mean_point[1]+10)]
    elif target == "cushion":
        search_space = floor[np.where((floor[:, 1]>mean_point[1]+2))]
    else:
        search_space = floor
    
    # find the nearest point to the mean point
    end_point = find_nearest_point(mean_point, search_space)

    start_point=[]
    step_size = 20
    cv2.circle(img, (int(end_point[0]), int(end_point[1])), 5, (0, 0, 0), -1)
    cv2.imshow("image", img)
    cv2.setMouseCallback("image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    RRT(img, img2,  start_point, end_point, step_size)