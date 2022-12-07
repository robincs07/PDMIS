import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import json
import argparse
import os
from tqdm import tqdm


# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "../apartment_0/habitat/mesh_semantic.ply"
path = "../apartment_0/habitat/info_semantic.json"

right_upper_corner = np.float32([-3.0716414, 0.76145476]).reshape(1,2)
right_down_corner = np.float32([4.25725, -4.600382]).reshape(1,2)
left_down_corner = np.float32([5.85725, 8.599619]).reshape(1,2)
print("right_upper_corner.shape: ", right_upper_corner.shape)

right_upper_pixel = np.float32([382, 78]).reshape(1,2)
right_down_pixel = np.float32([543, 342]).reshape(1,2)
left_down_pixel = np.float32([140, 403]).reshape(1,2)

sematic_id_map = {
    "refrigerator": 67,
    "rack": 66,
    "cushion": 29,
    "lamp": 47,
    "cooktop": 32
}

def warpAffine(points, Tran):
    """Apply Affine transformation to points
    Args:
        points: Nx2 matrix of (x,y) coordinates
        Tran: 2x3 Affine transformation matrix
    Returns:
        Nx2 matrix of transformed (x,y) coordinates
    """
    points = points.T
    points = np.vstack((points, np.ones((1, points.shape[1]))))
    points = np.dot(Tran, points)
    return points[:2, :].T

def compute_angle_cross(src_vec, dst_vec):
    """compute angle and cross vector between two vectors
    Args:
        src_vec: source vector
        dst_vec: destination vector
    Returns:
        angle: angle between two vectors
        cross_vec: cross vector between two vectors
    """
    angle = np.arccos(np.clip(np.dot(src_vec, dst_vec) / (np.linalg.norm(src_vec) * np.linalg.norm(dst_vec)), -1.0, 1.0))
    angle = angle * 180 / np.pi
    cross_vec = np.cross(src_vec, dst_vec)
    return angle, cross_vec



#global test_pic
#### instance id to semantic id 
with open(path, "r") as f:
    annotations = json.load(f)

id_to_label = []
instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])
for i in instance_id_to_semantic_label_id:
    if i < 0:
        id_to_label.append(0)
    else:
        id_to_label.append(i)
id_to_label = np.asarray(id_to_label)



########## initialize simulator ##########
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
    semantic_img = np.zeros((semantic_obs.shape[0], semantic_obs.shape[1], 3), dtype=np.uint8)
    mask_point = np.where(semantic_obs==sematic_id_map[args.target])
    semantic_img[mask_point] = np.asarray([0, 0, 255])
    return semantic_img

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]
    ##################################################################
    ### change the move_forward length or rotate angle
    ##################################################################
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.01) # 0.01 means 0.01 m
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=1.0) # 1.0 means 1 degree
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=1.0)
        ),
    }

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
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
print("#############################")
print("use keyboard to control the agent")
print(" w for go forward  ")
print(" a for turn left  ")
print(" d for trun right  ")
print(" f for finish and quit the program")
print("#############################")

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default='', help='target name')
args = parser.parse_args()

def navigateAndSee(action=""):
    if action in action_names:
        observations = sim.step(action)
        #print("action: ", action)

        rgb = transform_rgb_bgr(observations["color_sensor"])
        #cv2.imshow("depth", transform_depth(observations["depth_sensor"]))
        mask = transform_semantic(id_to_label[observations["semantic_sensor"]])
        fuse = cv2.addWeighted(rgb, 1.0, mask, 0.5, 0)
        cv2.imshow("result", fuse)
        # agent_state = agent.get_state()
        # sensor_state = agent_state.sensor_states['color_sensor']
        # print("camera pose: x y z rw rx ry rz")
        # print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)
        return fuse



# img and habitat corner coordinate
img_corner = np.concatenate((right_upper_pixel, right_down_pixel, left_down_pixel), axis=0)
habitat_corner = np.concatenate((right_upper_corner, right_down_corner, left_down_corner), axis=0)

# get the affine matrix
Trans = cv2.getAffineTransform(img_corner, habitat_corner)

# transform the image to habitat coordinate
path_points = np.load('path.npy')
path_points = np.delete(path_points, 0, axis=0)
habitat_xz = warpAffine(path_points, Trans)
habitat_path = np.zeros((habitat_xz.shape[0], 3))
habitat_path[:, 0] = habitat_xz[:, 0]
habitat_path[:, 2] = habitat_xz[:, 1]
# print("habitat_path: ", habitat_path)

RRT = cv2.imread('RRT_{}.png'.format(args.target)) # map image
cv2.imshow("RRT", RRT)
agent_state = habitat_sim.AgentState()
agent_state.position = habitat_path[0].T
agent.set_state(agent_state)
print("Agent set to start position!!!")

# for point in habitat_path:
#     #print("point: ", point)
#     agent_state = habitat_sim.AgentState()
#     agent_state.position = point.T
#     agent.set_state(agent_state)
#     navigateAndSee("move_forward")
#     cv2.waitKey(0)
action_set = []

for i in range(habitat_path.shape[0]-1):
    dst_vec = habitat_path[i+1] - habitat_path[i]
    # turn left or right
    if i == 0:
        angle, cross_vec = compute_angle_cross(np.array([0.0, 0.0, -0.01]), dst_vec)
    else:
        angle, cross_vec = compute_angle_cross(habitat_path[i] - habitat_path[i-1], dst_vec)
    if cross_vec[1] < 0: # turn right
        for i in range(int(angle)+1):
            action_set.append('turn_right')
    elif cross_vec[1] > 0: # turn left
        for i in range(int(angle)+1):
            action_set.append('turn_left')
    # move forward
    dist = np.linalg.norm(dst_vec) # distance
    dist = int(dist*100)
    for j in range(dist):
        action_set.append('move_forward')

output_dir = "../results"

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

print("Start making video!!!")
video_writer = cv2.VideoWriter(os.path.join(output_dir, "{}.mp4".format(args.target)), cv2.VideoWriter_fourcc(*'mp4v'), 90, (512, 512))
i = 0

for action in tqdm(action_set):
    i+=1
    frame = navigateAndSee(action)
    video_writer.write(frame)
    cv2.imshow("result", frame)
    cv2.waitKey(1)

video_writer.release()
print("finish!!!")




# while True:
#     keystroke = cv2.waitKey(0)
#     if keystroke == ord(FORWARD_KEY):
#         action = "move_forward"
#         navigateAndSee(action)
#         print("action: FORWARD")
#     elif keystroke == ord(LEFT_KEY):
#         action = "turn_left"
#         navigateAndSee(action)
#         print("action: LEFT")
#     elif keystroke == ord(RIGHT_KEY):
#         action = "turn_right"
#         navigateAndSee(action)
#         print("action: RIGHT")
#     elif keystroke == ord(FINISH):
#         print("action: FINISH")
#         break
#     else:
#         print("INVALID KEY")
#         continue
