from ast import Global
import numpy as np
from PIL import Image
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
from scipy.spatial.transform import Rotation
import quaternion as q
# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "../Homework1/apartment_0/habitat/mesh_semantic.ply"

f = open("GT_Pose.txt", "w")

points = []
Z_BEV = 0.0

Front_Camera_Pose = 0
BEV_Camera_Pose = 0
Front_camera_pose = np.array([0.4093778729438782, 0.12523484230041504, -0.6989918947219849, 0.6427876353263855, 0.0, -0.7660443782806396, 0.0
                              ])
BEV_camera_pose = np.array([0.4093778729438782, 1.125234842300415, -0.6989918947219849, 0.45451948046684265, -0.45451948046684265, -0.5416751503944397, -0.5416751503944397
                            ])


sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
}

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img


def transform_semantic(semantic_obs):
    semantic_img = Image.new(
        "P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img


def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    front_sensor_spec = habitat_sim.CameraSensorSpec()
    front_sensor_spec.uuid = "Front_Camera"
    front_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    front_sensor_spec.resolution = [settings["height"], settings["width"]]
    front_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    front_sensor_spec.orientation = [settings["sensor_pitch"], 0.0, 0.0]
    front_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    bev_sensor_spec = habitat_sim.CameraSensorSpec()
    bev_sensor_spec.uuid = "BEV_Camera"
    bev_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    bev_sensor_spec.resolution = [settings["height"], settings["width"]]
    bev_sensor_spec.position = [0.0, 2.5, Z_BEV]
    bev_sensor_spec.orientation = [-np.pi / 2, 0.0, 0.0]
    bev_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    # depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, 2.5, Z_BEV]
    depth_sensor_spec.orientation = [-np.pi / 2, 0.0, 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [
        front_sensor_spec,
        bev_sensor_spec,
        depth_sensor_spec,
    ]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)

# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([0.0, 0.0, 0.0])  # agent in world space
agent.set_state(agent_state)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(
    cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


def navigateAndSee(action=""):

    global BEV_Camera_Pose, Front_Camera_Pose
    if action in action_names:
        observations = sim.step(action)
        print("action: ", action)

        agent_state = agent.get_state()
        front_sensor_state = agent_state.sensor_states["Front_Camera"]

        print("front camera pose: x y z rw rx ry rz")
        print(
            front_sensor_state.position[0],
            front_sensor_state.position[1],
            front_sensor_state.position[2],
            front_sensor_state.rotation.w,
            front_sensor_state.rotation.x,
            front_sensor_state.rotation.y,
            front_sensor_state.rotation.z
        )
        Front_Camera_Pose = np.asarray([
            front_sensor_state.position[0],
            front_sensor_state.position[1],
            front_sensor_state.position[2],
            front_sensor_state.rotation.w,
            front_sensor_state.rotation.x,
            front_sensor_state.rotation.y,
            front_sensor_state.rotation.z]
        )
        f.write(
            f"{front_sensor_state.position[0]} {front_sensor_state.position[1]} {front_sensor_state.position[2]} {front_sensor_state.rotation.w} {front_sensor_state.rotation.x} {front_sensor_state.rotation.y} {front_sensor_state.rotation.z}\n"
        )

        # print BEV camera pose
        BEV_sensor_state = agent_state.sensor_states["BEV_Camera"]
        print("BEV camera pose: x y z rw rx ry rz")
        print(
            BEV_sensor_state.position[0],
            BEV_sensor_state.position[1],
            BEV_sensor_state.position[2],
            BEV_sensor_state.rotation.w,
            BEV_sensor_state.rotation.x,
            BEV_sensor_state.rotation.y,
            BEV_sensor_state.rotation.z
        )
        BEV_Camera_Pose = np.asarray([
            BEV_sensor_state.position[0],
            BEV_sensor_state.position[1],
            BEV_sensor_state.position[2],
            BEV_sensor_state.rotation.w,
            BEV_sensor_state.rotation.x,
            BEV_sensor_state.rotation.y,
            BEV_sensor_state.rotation.z]
        )
        f.write(
            f"{BEV_sensor_state.position[0]} {BEV_sensor_state.position[1]} {BEV_sensor_state.position[2]} {BEV_sensor_state.rotation.w} {BEV_sensor_state.rotation.x} {BEV_sensor_state.rotation.y} {BEV_sensor_state.rotation.z}\n"
        )
        cv2.imshow("BEV", transform_rgb_bgr(observations["BEV_Camera"]))
        cv2.imshow("Front View", transform_rgb_bgr(
            observations["Front_Camera"]))
        # RGB_Image shape: (512, 512, 3), Depth_Image shape: (512, 512, 1), Semantic_Image shape: (512, 512, 3)
        # print(RGB_Image.shape)
        # print(Depth_Image.shape)
        # print(Semantic_Image.shape)

        cv2.imwrite("BEV.png", transform_rgb_bgr(observations["BEV_Camera"]))
        cv2.imwrite("front.png", transform_rgb_bgr(
            observations["Front_Camera"]))
        cv2.imwrite("depth.png", transform_depth(observations["depth_sensor"]))


class Projection(object):
    def __init__(self, image_path, points):
        """
        :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
        Project the top view pixels to the front view pixels.
        :return: New pixels on perspective(front) view image
        """

        ### TODO ###

        # intrinsic
        K = intrinsic_from_fov(512, 512)
        Points_3d = depth_image_to_point_cloud(np.linalg.inv(K), points)
        print("K: ", K)
        Homo_Points_3d = np.ones((4, Points_3d.shape[1]))
        Homo_Points_3d[:3, :] = Points_3d

        # compute top to front rotation matrix
        Top_To_Global_R = get_global_rotation(BEV_Camera_Pose)
        Global_To_Front_R = np.linalg.inv(
            get_global_rotation(Front_Camera_Pose))

        print("!!!!!!!\n", Top_To_Global_R)
        print(get_global_rotation(Front_Camera_Pose))
        Top_To_Front_R = Global_To_Front_R@Top_To_Global_R
        Top_To_Front_Trans = np.identity(4)
        Top_To_Front_Trans[:3, :3] = Top_To_Front_R
        Top_To_Front_Trans[:3, 3] = Front_Camera_Pose[:3]-BEV_Camera_Pose[:3]
        # Top_To_Front_Trans[0, 3] *= (-1)
        # Top_To_Front_Trans[2, 3] *= (-1)

        print("Top to front trans: \n", Top_To_Front_Trans)

        # project to the front view
        Homo_Points_3d = Top_To_Front_Trans@Homo_Points_3d
        print("HOmo_3d:\n", Homo_Points_3d)
        Homo_Points_2d = K[:3, :3]@Homo_Points_3d[:3, :]
        print("Homo_points_2d:\n", Homo_Points_2d)
        New_Points = Homo_Points_2d/Homo_Points_3d[2]
        print("New_Points: \n", New_Points)
        return New_Points[:2, :].T

    def show_image(
        self, new_pixels, img_name="projection.png", color=(0, 0, 255), alpha=0.4
    ):
        """
        Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.int32(np.array(new_pixels))], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(f"Top to front view projection {img_name}", new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, " ", y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(top_img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("image", top_img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, " ", y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = top_img[y, x, 0]
        g = top_img[y, x, 1]
        r = top_img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow("image", top_img)


def get_global_rotation(Position):
    R_Matrix = q.as_rotation_matrix(q.as_quat_array([Position[3:7]]))
    Trans = np.zeros((4, 4), dtype=np.float32)
    Trans[0:3, 0:3] = R_Matrix
    Trans[0:3, 3] = Position[:3].T
    Trans[3, 3] = 1
    return R_Matrix


def intrinsic_from_fov(Height, Width, Fov=90):

    Px, Py = (Width / 2, Height / 2)
    Xfov = Fov / 360.0 * 2.0 * np.pi
    Fx = Width / (2.0 * np.tan(Xfov / 2.0))

    Yfov = 2.0 * np.arctan(np.tan(Xfov / 2) * Height / Width)
    Fy = Height / (2.0 * np.tan(Yfov / 2.0))

    return np.array(
        [[Fx, 0, Px, 0.0],
         [0, Fy, Py, 0.0],
         [0, 0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    )


def depth_image_to_point_cloud(K_Inv, points):

    Depth_Image = cv2.imread(
        "depth.png",
        cv2.IMREAD_ANYDEPTH)
    print("depth Image: ", Depth_Image.shape)

    points = np.asarray(points)
    Dep_Val = []
    for i in range(points.shape[0]):
        Dep_Val.append(Depth_Image[points[i][0], points[i][1]])
    print(Dep_Val)
    Dep_Val = np.asarray(Dep_Val)
    Homo_Points = np.ones((3, points.shape[0]))
    Homo_Points[:2, :] = points.T

    Camera_Coords = K_Inv[:3, :3]@Homo_Points*Dep_Val/255*10

    return Camera_Coords


def pixel_coord_np(Height, Width):
    # create the matrix of every pixel coordinate
    X = np.linspace(0, Width - 1, Width).astype(int)
    Y = np.linspace(0, Height - 1, Height).astype(int)
    [X, Y] = np.meshgrid(X, Y)
    return np.vstack((X.flatten(), Y.flatten(), np.ones_like(X.flatten())))


FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH = "f"
print("#############################")
print("use keyboard to control the agent")
print(" w for go forward  ")
print(" a for turn left  ")
print(" d for trun right  ")
print(" f for finish and quit the program")
print("#############################")

if __name__ == "__main__":
    action = "move_forward"
    navigateAndSee(action)

    while True:
        keystroke = cv2.waitKey(0)
        if keystroke == ord(FORWARD_KEY):
            action = "move_forward"
            navigateAndSee(action)
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = "turn_left"
            navigateAndSee(action)
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = "turn_right"
            navigateAndSee(action)
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            print("action: FINISH")
            f.close()
            break
        else:
            print("INVALID KEY")
            continue

    cv2.destroyAllWindows()
    pitch_ang = -90
    print("Front Camera Pose: ", Front_Camera_Pose)
    print("BEV Camera Pose: ", BEV_Camera_Pose)

    front_rgb = "front.png"
    top_rgb = "BEV.png"

    # click the pixels on window
    top_img = cv2.imread(top_rgb)
    cv2.imshow("image", top_img)
    cv2.setMouseCallback("image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # print front camera pose

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels)

    # projection = Projection(front_rgb, points)
    # new_pixels = projection.top_to_front(theta=pitch_ang)
    # projection.show_image(new_pixels)
