import numpy as np
from PIL import Image
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import open3d as o3d
import copy
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors

VOXEL_SIZE = 0.002
NUM_OF_PHOTOS = 204
PCD_NAME = "test_true.pcd"
USE_OPEN3D = True


def get_global_trans_matrix():
    with open("Data_collection/first_floor/GT_Pose.txt", "r") as f:
        line = f.readline()
        Position = line.split()
        R_Matrix = Rotation.from_quat([Position[3:7]]).as_matrix().squeeze()
        T_Matrix = np.asarray(Position[0:3]).transpose()
        Trans = np.zeros((4, 4), dtype=np.float32)
        Trans[0:3, 0:3] = R_Matrix
        Trans[0:3, 3] = T_Matrix
        Trans[3, 3] = 1
        f.close()
        return Trans


def depth_image_to_point_cloud(Photo_Num, K_Inv):

    # Read rgb and depth image
    Rgb_Image = cv2.cvtColor(
        cv2.imread("Data_collection/first_floor/rgb/%d.png" % Photo_Num),
        cv2.COLOR_BGR2RGB)
    Depth_Image = cv2.imread(
        "Data_collection/first_floor/depth/%d.png" % Photo_Num,
        cv2.IMREAD_ANYDEPTH)

    Height, Width, _ = Rgb_Image.shape

    Pixel_Coords = pixel_coord_np(Width, Height)
    Camera_Coords = K_Inv[:3, :3] @ Pixel_Coords * Depth_Image.flatten() / 1000

    Rgb_np = np.asarray(Rgb_Image, dtype=np.float32).reshape(Height * Width, 3)
    Rgb_np /= 255

    Pcd_Cam = o3d.geometry.PointCloud()
    Pcd_Cam.points = o3d.utility.Vector3dVector(Camera_Coords.T[:, :3])
    Points = np.asarray(Pcd_Cam.points)
    Pcd_Cam.colors = o3d.utility.Vector3dVector(Rgb_np)
    Pcd_Cam = Pcd_Cam.select_by_index(np.where(Points[:, 2] < 0.08)[0])

    # flip
    # Pcd_Cam.transform([[1, 0, 0, 0],
    #                    [0, -1, 0, 0],
    #                    [0, 0, -1, 0],
    #                    [0, 0, 0, 1]])

    # visualize
    # o3d.visualization.draw_geometries([Pcd_Cam])

    return Pcd_Cam


def pixel_coord_np(Height, Width):
    # create the matrix of every pixel coordinate
    X = np.linspace(0, Width - 1, Width).astype(int)
    Y = np.linspace(0, Height - 1, Height).astype(int)
    [X, Y] = np.meshgrid(X, Y)
    return np.vstack((X.flatten(), Y.flatten(), np.ones_like(X.flatten())))


def intrinsic_from_fov(Height, Width, Fov=90):

    Px, Py = (Width / 2, Height / 2)
    Xfov = Fov / 360.0 * 2.0 * np.pi
    Fx = Width / (2.0 * np.tan(Xfov / 2.0))

    Yfov = 2.0 * np.arctan(np.tan(Xfov / 2) * Height / Width)
    Fy = Height / (2.0 * np.tan(Yfov / 2.0))

    return np.array([[Fx, 0, Px, 0.0], [0, Fy, Py, 0.0], [0, 0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def local_icp_algorithm(Source, Target):
    Threshold = VOXEL_SIZE * 0.4
    if USE_OPEN3D:
        Global_Trans = global_registration_algorithm(Source=Source,
                                                     Target=Target,
                                                     Voxel_Size=VOXEL_SIZE)
        Reg_P2l = o3d.pipelines.registration.registration_icp(
            Source, Target, Threshold, Global_Trans.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        # print(Reg_P2l)
        # print(f"Transformation is: {Reg_P2l.transformation}")
        return Reg_P2l.transformation
    else:
        Global_Trans = global_registration_algorithm(Source=Source,
                                                     Target=Target,
                                                     Voxel_Size=VOXEL_SIZE)

        tgt = np.asarray(Target.points)
        src = np.asarray(Source.points)

        Homo_Source = np.ones((4, src.shape[0]))
        Homo_Target = np.ones((4, tgt.shape[0]))

        Homo_Source[:3, :] = src.T
        Homo_Target[:3, :] = tgt.T

        Homo_Source = Global_Trans.transformation @ Homo_Source
        All_Trans = Global_Trans.transformation
        src = Homo_Source

        Prev_Error = 0

        for i in range(20):

            Distances, Indices = find_near_neigh(Src=Homo_Source[:3, :].T,
                                                 Tgt=Homo_Target[:3, :].T)

            Condition = np.where(Distances > 1.5 * VOXEL_SIZE)
            Indices = np.delete(Indices, Condition)
            Homo_Target = Homo_Target[:, Indices]
            Homo_Source = np.delete(Homo_Source.T, Condition, axis=0).T

            Now_Trans = get_fit_transform(Homo_Source[:3, :].T,
                                          Homo_Target[:3, :].T)

            Homo_Source = Now_Trans @ Homo_Source
            All_Trans = Now_Trans @ All_Trans

            Mean_Error = np.sum(Distances) / Distances.size
            if Prev_Error - Mean_Error < 0.000001:
                break

        return All_Trans


def find_near_neigh(Src, Tgt):
    Near_Neigh = NearestNeighbors(n_neighbors=1)
    Near_Neigh.fit(Tgt)
    Distances, Indices = Near_Neigh.kneighbors(Src, return_distance=True)
    return Distances.ravel(), Indices.ravel()


def get_fit_transform(Source, Target):

    Center_Source = np.mean(Source, axis=0)
    Center_Target = np.mean(Target, axis=0)
    # print(f"Center_Source: {Center_Source}")
    # print(f"Center_Target: {Center_Target}")

    Source_pl = Source - Center_Source
    Target_pl = Target - Center_Target

    # W = Source_pl.T@Target_pl
    # U, _, Vh = np.linalg.svd(W)
    # R = Vh.T@U.T

    W = Target_pl.T @ Source_pl
    U, _, Vh = np.linalg.svd(W)
    R = U @ Vh

    if np.linalg.det(R) < 0:
        print("!!!")
        Vh[2, :] *= -1
        R = U @ Vh

    T = Center_Target - (R @ Center_Source)

    Trans = np.identity(4)
    Trans[:3, :3] = R
    Trans[:3, 3] = T

    return Trans


def global_registration_algorithm(Source, Target, Voxel_Size):

    Source_Down, Source_Fpfh = preprocess_point_cloud(Source, Voxel_Size)
    Target_Down, Target_Fpfh = preprocess_point_cloud(Target, Voxel_Size)
    Distance_Threshold = Voxel_Size
    Result_RANSAC = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        Source_Down, Target_Down, Source_Fpfh, Target_Fpfh, True,
        Distance_Threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                Distance_Threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.95))
    # print(Result_RANSAC)
    # draw_registration_result(Source, Target, Result_RANSAC.transformation)

    return Result_RANSAC


def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature,
                                             max_nn=100))
    return pcd_down, pcd_fpfh


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp + target_temp])


def get_pcd_remove_ceiling(Point_Cloud):

    Points = np.asarray(Point_Cloud.points)
    Mean_Y = np.mean(Points, axis=0)[1]
    Temp_Point_Cloud = Point_Cloud.select_by_index(
        np.where(Points[:, 1] < Mean_Y)[0])

    return Temp_Point_Cloud


def main():

    # Read image and compute K_inv
    Rgb_Image = cv2.cvtColor(
        cv2.imread("Data_collection/first_floor/rgb/1.png"), cv2.COLOR_BGR2RGB)
    Height, Width, _ = Rgb_Image.shape
    K = intrinsic_from_fov(Height, Width, 90)
    K_Inv = np.linalg.inv(K)

    # Create Point cloud
    Point_Cloud_List = []
    print("Creating Point Cloud:")
    for Photo_Num in tqdm(range(1, NUM_OF_PHOTOS + 1)):
        Point_Cloud = depth_image_to_point_cloud(Photo_Num, K_Inv)
        Point_Cloud_List.append(Point_Cloud)

    # append transformation matrix of cam_2 coordinate to cam_1 coordiante to list
    Camera1_Trans_List = []
    Previous_Trans_List = []
    Trans = local_icp_algorithm(Source=Point_Cloud_List[1],
                                Target=Point_Cloud_List[0])
    Camera1_Trans_List.append(Trans)
    Previous_Trans_List.append(Trans)

    # Compute the extrinsic matrix to previous camera coordinate
    print("Compute Extrinsic Matrix:")
    for i in tqdm(range(1, NUM_OF_PHOTOS - 1), leave=True):
        Trans = local_icp_algorithm(Source=Point_Cloud_List[i + 1],
                                    Target=Point_Cloud_List[i])
        Camera1_Trans_List.append(Camera1_Trans_List[i - 1] @ Trans)
        Previous_Trans_List.append(Trans)

    # Align the point cloud
    print("Align Point Cloud:")
    for i in tqdm(range(1, NUM_OF_PHOTOS)):
        Point_Cloud_List[i].transform(Camera1_Trans_List[i - 1])

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.01, origin=[0, 0, 0])

    All_Point_Cloud = Point_Cloud_List[0]
    for i in range(1, NUM_OF_PHOTOS):
        All_Point_Cloud += Point_Cloud_List[i]

    # trasfrom to gloabal coordinate
    All_Point_Cloud.transform((get_global_trans_matrix()))
    Point_Cloud_Without_Ceiling = get_pcd_remove_ceiling(All_Point_Cloud)

    # Read Position file , Adjust the line point and save it to list
    Position_List = []
    Cam_1_Pose = []
    with open("Data_collection/first_floor/GT_Pose.txt", "r") as f:
        for f_idx, line in enumerate(f):
            Position = line.split()

            for idx, item in enumerate(Position):
                item = float(item)
                Position[idx] = item

            if f_idx == 0:
                Cam_1_Pose = Position

            for i in range(0, 3):
                Position[i] = ((Position[i] - Cam_1_Pose[i]) / 10 * 255 /
                               1000) + Cam_1_Pose[i]

            Position_List.append(Position[0:3])

    # Create line set of ground truth trajectory
    Line_List = []
    for i in range(len(Position_List) - 1):
        Line_List.append([i, i + 1])
    Line_Set_GT = o3d.geometry.LineSet()
    Line_Set_GT.points = o3d.utility.Vector3dVector(
        Position_List[:NUM_OF_PHOTOS])
    Line_Set_GT.lines = o3d.utility.Vector2iVector(Line_List[:NUM_OF_PHOTOS -
                                                             1])

    # Create line of estimated camera pose trajectory
    Estimate_Camera_Pose = []
    Estimate_Camera_Pose.append(Position_List[0])
    Camera1_To_Global = get_global_trans_matrix()
    for idx, matrix in enumerate(Camera1_Trans_List):
        Estimate_Camera_Pose.append((Camera1_To_Global @ matrix[:4, 3])[:3])
    # print("lenth: ", len(Estimate_Camera_Pose))
    # Draw the estimate trajectory
    Line_Set_Estimate = o3d.geometry.LineSet()
    Line_Set_Estimate.points = o3d.utility.Vector3dVector(
        Estimate_Camera_Pose[:NUM_OF_PHOTOS])
    Line_Set_Estimate.lines = o3d.utility.Vector2iVector(
        Line_List[:NUM_OF_PHOTOS - 1])
    Line_Set_Estimate.paint_uniform_color([0, 0, 1])

    # Visualize
    o3d.visualization.draw_geometries(
        [Point_Cloud_Without_Ceiling, Line_Set_GT, Line_Set_Estimate])

    o3d.io.write_point_cloud(PCD_NAME, Point_Cloud_Without_Ceiling)


if __name__ == "__main__":
    main()
