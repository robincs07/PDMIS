# Perception and Decision Making in Intelligent Systems Homework 1 Report

## **Task 1:**

1. **Code**
    1. First, walk through the scene and find where I want to save the photos of front view and BEV view.

        ```python
        cv2.imshow("BEV", transform_rgb_bgr(observations["BEV_Camera"]))
        cv2.imshow("Front View", transform_rgb_bgr(observations["Front_Camera"]))
        cv2.imwrite("BEV.png", transform_rgb_bgr(observations["BEV_Camera"]))
        cv2.imwrite("front.png", transform_rgb_bgr(observations["Front_Camera"]))
        cv2.imwrite("depth.png", transform_depth(observations["depth_sensor"]))
        ```

    2. After clicking the points on the BEV image, I compute the intrinsic matrix and apply to the points.

        ```python
        K = intrinsic_from_fov(512, 512)
        Points_3d = depth_image_to_point_cloud(np.linalg.inv(K), points)
        print("K: ", K)
        Homo_Points_3d = np.ones((4, Points_3d.shape[1]))
        Homo_Points_3d[:3, :] = Points_3d
        ```

    3. We need to compute **two rotation matrix** to project BEV view →front view:
        1. BEV →Global
        2. Global →Front

        then multiply two matrix

        ```python
        Top_To_Global_R = get_global_rotation(BEV_Camera_Pose)
        Global_To_Front_R = np.linalg.inv(get_global_rotation(Front_Camera_Pose))
        Top_To_Front_R = Global_To_Front_R@Top_To_Global_R
        ```

    4. Compute the BEV →front  translation matrix, just subtract two camera’s position. Then, concatenate the rotation and translation matrix to form the transformation matrix.

        ```python
        Top_To_Front_Trans = np.identity(4)
        Top_To_Front_Trans[:3, :3] = Top_To_Front_R
        Top_To_Front_Trans[:3, 3] = Front_Camera_Pose[:3]-BEV_Camera_Pose[:3]
        ```

    5. Finally, multiply transformation matrix and point then multiply the intrinsic matrix to get the pixel in front view.

    ```python
    Homo_Points_3d = Top_To_Front_Trans@Homo_Points_3d
    Homo_Points_2d = K[:3, :3]@Homo_Points_3d[:3, :]
    New_Points = Homo_Points_2d/Homo_Points_3d[2]
    ```

2. **Result and Discussion**
    1. **Result of projection**

        ![Untitled](Perception%20and%20Decision%20Making%20in%20Intelligent%20Syst%20febd6bf0651346d6899be49cb84e460f/Untitled.png)

        ![Untitled](Perception%20and%20Decision%20Making%20in%20Intelligent%20Syst%20febd6bf0651346d6899be49cb84e460f/Untitled%201.png)

        ![Untitled](Perception%20and%20Decision%20Making%20in%20Intelligent%20Syst%20febd6bf0651346d6899be49cb84e460f/Untitled%202.png)

        ![Untitled](Perception%20and%20Decision%20Making%20in%20Intelligent%20Syst%20febd6bf0651346d6899be49cb84e460f/Untitled%203.png)

## Task 2

    1. Code
        1. First, save the RGB image, Depth image and camera position.

            ```python
            f.write(f"{sensor_state.position[0]} {sensor_state.position[1]} {sensor_state.position[2]} {sensor_state.rotation.w} {sensor_state.rotation.x} {sensor_state.rotation.y} {sensor_state.rotation.z}\n")
            # RGB list length: [512][512][3]
            cv2.imwrite(
                f"{IMAGE_PATH}/rgb/%d.png" % Photo_Num,
                transform_rgb_bgr(observations["color_sensor"]),
            )
            cv2.imwrite(
                f"{IMAGE_PATH}/depth/%d.png" % Photo_Num,
                transform_depth(observations["depth_sensor"]),
            )
            cv2.imwrite(
                f"{IMAGE_PATH}/semantic/%d.png" % Photo_Num,
                transform_semantic(observations["semantic_sensor"]),
            )
            ```

        2. Next, compute intrinsic matrix, then create point clouds from depth image and RGB image.
            1. Compute intrinsic matrix via two formulation:

                ![Untitled](Perception%20and%20Decision%20Making%20in%20Intelligent%20Syst%20febd6bf0651346d6899be49cb84e460f/Untitled%204.png)

                ![Untitled](Perception%20and%20Decision%20Making%20in%20Intelligent%20Syst%20febd6bf0651346d6899be49cb84e460f/Untitled%205.png)

                ```python
                def intrinsic_from_fov(Height, Width, Fov=90):
                
                    Px, Py = (Width / 2, Height / 2)
                    Xfov = Fov / 360.0 * 2.0 * np.pi
                    Fx = Width / (2.0 * np.tan(Xfov / 2.0))
                
                    Yfov = 2.0 * np.arctan(np.tan(Xfov / 2) * Height / Width)
                    Fy = Height / (2.0 * np.tan(Yfov / 2.0))
                
                    return np.array([[Fx, 0, Px, 0.0],
                           [0, Fy, Py, 0.0],
                           [0, 0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0, 1.0]])
                ```

            2. Create point clouds

                ```python
                def depth_image_to_point_cloud(Photo_Num, K_Inv):
                    # Read rgb and depth image
                    Rgb_Image = cv2.cvtColor(
                        cv2.imread(f"{IMAGE_PATH}/rgb/%d.png" % Photo_Num),
                        cv2.COLOR_BGR2RGB)
                    Depth_Image = cv2.imread(
                        f"{IMAGE_PATH}/depth/%d.png" % Photo_Num,
                        cv2.IMREAD_ANYDEPTH)
                  # Get height and width of image
                    Height, Width, _ = Rgb_Image.shape
                  # Compute 3D position of every pixels in image
                    Pixel_Coords = pixel_coord_np(Width, Height)
                    Camera_Coords = K_Inv[:3, :3] @ Pixel_Coords * Depth_Image.flatten() / 1000
                  # Get every points' RGB value
                    Rgb_np = np.asarray(Rgb_Image, dtype=np.float32).reshape(Height * Width, 3)
                    Rgb_np /= 255
                  # Create point cloud
                    Pcd_Cam = o3d.geometry.PointCloud()
                    Pcd_Cam.points = o3d.utility.Vector3dVector(Camera_Coords.T[:, :3])
                    Points = np.asarray(Pcd_Cam.points)
                    Pcd_Cam.colors = o3d.utility.Vector3dVector(Rgb_np)
                    Pcd_Cam = Pcd_Cam.select_by_index(np.where(Points[:, 2] < 0.08)[0])
                    # Flip
                    Pcd_Cam.transform([[1, 0, 0, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, -1, 0],
                                       [0, 0, 0, 1]])
                    return Pcd_Cam
                ```

        3. Applying ICP algorithm to align every point clouds.There are two versions of the ICP algorithm (Open3D and Mine).
            1. Open3D

                Same as the Open3D official tutorial. Do the global registration (RANSAC) first, then apply the local registration (point to plane).

                ```python
                # global registration
                Global_Trans, Source_down, Target_down = global_registration_algorithm(Source=Source,
                                                                                       Target=Target,
                                                                                       Voxel_Size=VOXEL_SIZE)
                # local registration
                Source_down.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
                Target_down.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
                Local_reg = o3d.pipelines.registration.registration_icp(
                    Source_down, Target_down, Threshold, Global_Trans.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
                ```

            2. Mine
                1. Doing the global registration first.
                2. Finding target point which is nearest point source points via nearest neighbors algorithm.
                3. Deleting the points which the distance from target point to source points > threshold.
                4. Computing the transformation matrix by Umeyama algorithm
                5. Repeating the step b ~ d until the error converge.

                ```python
                Global_Trans, Source_down, Target_down = global_registration_algorithm(Source=Source,
                                                                                               Target=Target,
                                                                                               Voxel_Size=VOXEL_SIZE)
                
                tgt = np.asarray(Target_down.points)
                src = np.asarray(Source_down.points)
                Homo_Source = np.ones((4, src.shape[0]))
                Homo_Target = np.ones((4, tgt.shape[0]))
                Homo_Source[:3, :] = src.T
                Homo_Target[:3, :] = tgt.T
                Homo_Source = Global_Trans.transformation @ Homo_Source
                All_Trans = Global_Trans.transformation
                Prev_Error = 0
                for i in range(100):
                  # find nearest correspondece point
                    Distances, Indices = find_near_neigh(Src=Homo_Source[:3, :].T,
                                                         Tgt=Homo_Target[:3, :].T)
                
                    temp_source = Homo_Source
                    temp_target = Homo_Target
                    Condition = np.where(Distances > VOXEL_SIZE)
                    Indices = np.delete(Indices, Condition)
                    temp_target = temp_target[:, Indices]
                    temp_source = np.delete(temp_source.T, Condition, axis=0).T
                    Now_Trans = get_fit_transform(temp_source[:3, :].T,
                                                  temp_target[:3, :].T)
                    Homo_Source = Now_Trans @ Homo_Source
                    All_Trans = Now_Trans @ All_Trans
                    Mean_Error = np.sum(Distances) / Distances.size
                    if np.abs(Prev_Error - Mean_Error) < 0.00000001:
                        break
                    Prev_Error = Mean_Error
                return All_Trans
                ```

        4. Through above step, I will get the transformation matrix from camera_i+1 to camera_i. Since I want to let every points cloud in same coordinate, so I multiply the transformation matrix to get the transformation matrix from camera_i+1 to camera_1.

            ```python
            Camera1_Trans_List = []
            Camera1_Trans_List.append(np.identity(4))
            print("Compute Extrinsic Matrix:")
            for i in tqdm(range(0, NUM_OF_PHOTOS-1), leave=True):
                if i == 0:
                    temp_pointcloud = Point_Cloud_List[0]
                else:
                    temp_pointcloud = Point_Cloud_List[i] + Point_Cloud_List[i-1]
                Trans = local_icp_algorithm(Source=Point_Cloud_List[i + 1],
                                            Target=temp_pointcloud)
                Point_Cloud_List[i+1].transform(Trans)
                Camera1_Trans_List.append(Trans)
            ```

        5. Last, transform all point cloud to global coordinate and remove the ceiling.

            ```python
            # add all point cloud to one point cloud
            All_Point_Cloud = Point_Cloud_List[0]
            for i in range(1, NUM_OF_PHOTOS):
                All_Point_Cloud += Point_Cloud_List[i]
            # trasfrom to gloabal coordinate
            All_Point_Cloud.transform((get_global_trans_matrix()))
            Point_Cloud_Without_Ceiling = get_pcd_remove_ceiling(All_Point_Cloud)
            ```

        6. Create the ground truth trajectory by GT_Pose.txt and the estimation trajectory by translation part of transformation matrix.

            ```python
            # Create line set of ground truth trajectory
            Line_List = []
            for i in range(len(Position_List) - 1):
                Line_List.append([i, i + 1])
            Line_Set_GT = o3d.geometry.LineSet()
            Line_Set_GT.points = o3d.utility.Vector3dVector(Position_List[:NUM_OF_PHOTOS])
            Line_Set_GT.lines = o3d.utility.Vector2iVector(Line_List[:NUM_OF_PHOTOS - 1])
            # Create line of estimated camera pose trajectory
            Estimate_Camera_Pose = []
            Estimate_Camera_Pose.append(Position_List[0])
            Camera1_To_Global = get_global_trans_matrix()
            for idx, matrix in enumerate(Camera1_Trans_List):
                Estimate_Camera_Pose.append((Camera1_To_Global @ matrix[:4, 3])[:3])
            ```

    2. **Result and Discussion**
        1. 1F Open3D ICP

            ![Untitled](Perception%20and%20Decision%20Making%20in%20Intelligent%20Syst%20febd6bf0651346d6899be49cb84e460f/Untitled%206.png)

            MSE: 0.000306 (m)

        2. 1F my ICP

            ![Untitled](Perception%20and%20Decision%20Making%20in%20Intelligent%20Syst%20febd6bf0651346d6899be49cb84e460f/Untitled%207.png)

            MSE: 0.001732 (m)

        3. 2F Open3D ICP

            ![Untitled](Perception%20and%20Decision%20Making%20in%20Intelligent%20Syst%20febd6bf0651346d6899be49cb84e460f/Untitled%208.png)

            MSE: 0.0001157 (m)

        4. 2F my ICP

            ![Untitled](Perception%20and%20Decision%20Making%20in%20Intelligent%20Syst%20febd6bf0651346d6899be49cb84e460f/Untitled%209.png)

            MSE: 0.000311 (m)

        5. Mine ICP is not performance as good as the Open3D’s ICP. I think the main difference is the method. I use the point to point method, and the Open3D use the point to plane method. Although I have used some method like if the distance between source point correspond to target is too large I will delete the point. This method has a great impact on aligning the point cloud but still cannot as good as Open3D version.
