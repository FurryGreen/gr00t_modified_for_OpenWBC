import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Array, Lock
from multiprocessing import Process, Queue
import threading

import zmq
import time
import json

import os 
import sys
import copy
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

from inference_deploys.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from inference_deploys.robot_control.robot_hand_unitree import Dex3_1_Controller, Gripper_Controller
from inference_deploys.image_server.image_client import ImageClient

import pickle
import socket



data_lock = threading.Lock()

def zmq_listener():
    global infer_flag
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://192.168.123.164:5557")
    socket.setsockopt_string(zmq.SUBSCRIBE, "infer_start")

    while True:
        msg = socket.recv_string()
        topic, data_json = msg.split(' ', 1)
        infer_msg = int(json.loads(data_json))

        # 更新共享变量（带锁）
        with data_lock:
            infer_flag = infer_msg
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type = str, default = './utils/data', help = 'path to save data')
    parser.add_argument('--frequency', type = int, default = 30.0, help = 'save data\'s frequency')

    parser.add_argument('--record', action = 'store_true', help = 'Save data or not')
    parser.add_argument('--no-record', dest = 'record', action = 'store_false', help = 'Do not save data')
    parser.set_defaults(record = False)

    parser.add_argument('--arm', type=str, default = 'G1_29', choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--hand', type=str, default = 'dex3', choices=['dex3', 'gripper', 'inspire1'], help='Select hand controller')
    
    
    #### for policy
    parser.add_argument('--goal', type=str, default='pick_pink_fox', help='Language Goal.') ## TODO: 带下划线...?
    parser.add_argument('--model-path', type=str, default='./save/multiobj_pick_WBC', help='Path to the model checkpoint directory.')
    parser.add_argument('--embodiment-tag', type=str, default='new_embodiment', help='The embodiment tag for the model.')
    parser.add_argument('--data-config', type=str, default='openwbc_g1', help='The name of the data config to use.')
    parser.add_argument('--server', action='store_true', help='Whether to run the server.')
    parser.add_argument('--client', action='store_true', help='Whether to run the client.')
    parser.add_argument('--denoising-steps', type=int, default=4, help='The number of denoising steps to use.')
    parser.add_argument('--action_horizon', type=int, default=16, help='The action horizon for the policy.')
    

    args = parser.parse_args()
    print(f"args:{args}\n")
    
    ### Policy initialization
    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )
    
    modality_config = policy.get_modality_config()
    ### action.left_hand --> left_hand
    modality_keys = [_.split('.')[-1] for _ in modality_config['action'].modality_keys]

    print('modality keys:', modality_keys)
    
    arm_keys = ['left_arm', 'right_arm']
    base_key = ['base_motion']

    # image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
    img_config = {
        'fps':30,                                                          # frame per second
        'head_camera_type': 'realsense',                                  # opencv or realsense
        'head_camera_image_shape': [480, 640],                            # Head camera resolution  [height, width]
        'head_camera_id_numbers': ["218622271739"],                       # realsense camera's serial number
        # 'wrist_camera_type': 'opencv', 
        # 'wrist_camera_image_shape': [480, 640],                           # Wrist camera resolution  [height, width]
        # 'wrist_camera_id_numbers': [0,1],                                 # '/dev/video0' and '/dev/video1' (opencv)
    }

    publisher_thread = threading.Thread(target=zmq_listener, daemon=True)
    publisher_thread.start()
    
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5556")  # 绑定到端口

    topic = "action_cmd" ### TODO: 这里改成发送action command


    ASPECT_RATIO_THRESHOLD = 2.0 # If the aspect ratio exceeds this value, it is considered binocular
    if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    if 'wrist_camera_type' in img_config:
        WRIST = True
    else:
        WRIST = False
    
    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = tv_img_shm.buf)

    if WRIST:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                 wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name)
    else:
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name)

    image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()


    # arm
    if args.arm == 'G1_29':
        arm_ctrl = G1_29_ArmController()
    elif args.arm == 'G1_23':
        arm_ctrl = G1_23_ArmController()
    elif args.arm == 'H1_2':
        arm_ctrl = H1_2_ArmController()
    elif args.arm == 'H1':
        arm_ctrl = H1_ArmController()

    # hand
    if args.hand == "dex3":
        left_hand_array = Array('d', 75, lock = True)         # [input]
        right_hand_array = Array('d', 75, lock = True)        # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock = False)  # [output] current left, right hand state(14) data.
        dual_hand_action_array = Array('d', 14, lock = False) # [output] current left, right hand action(14) data.
        hand_ctrl = Dex3_1_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
    elif args.hand == "gripper":
        left_hand_array = Array('d', 75, lock=True)
        right_hand_array = Array('d', 75, lock=True)
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
        dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
        gripper_ctrl = Gripper_Controller(left_hand_array, right_hand_array, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array)
    else:
        pass
    

    try:
        user_input = input("Please enter the start signal (enter 't' to start the Gr00t policy):\n")
        if user_input.lower() == 't':
            arm_ctrl.speed_gradual_max()

            running = True
            total_start_time = time.time()
            current_flag = True
            policy_start = False
            count = 0
            #### 这里开始可以包装为函数：model_infer_mode
            while running:
                
                if policy_start: #### 第一次先发出
                    with data_lock:
                        current_flag = infer_flag
                else:
                    policy_start = True
                    
                if current_flag:
                    start_time = time.time()                  

                    # get current state data.
                    current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
                    current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
                    current_lr_leg_q = arm_ctrl.get_current_dual_arm_q()
            
                    left_arm_state  = current_lr_arm_q[:7]
                    right_arm_state = current_lr_arm_q[-7:]
                    left_leg_state = current_lr_leg_q[0:6]
                    right_leg_state = current_lr_leg_q[6:12]
                    # dex hand or gripper
                    if args.hand == "dex3":
                        with dual_hand_data_lock:
                            left_hand_state = dual_hand_state_array[:7]
                            right_hand_state = dual_hand_state_array[-7:]

                    elif args.hand == "gripper":
                        with dual_gripper_data_lock:
                            left_hand_state = [dual_gripper_state_array[1]]
                            right_hand_state = [dual_gripper_state_array[0]]

                    obs = {
                        "video.ego_view": copy.deepcopy(tv_img_array), ### important: COPY!!!
                        "state.left_arm": left_arm_state.reshape(1, -1),
                        "state.right_arm": right_arm_state.reshape(1, -1),
                        "state.left_hand": left_hand_state.reshape(1, -1) if args.hand == "dex3" else np.zeros((1, 6)),
                        "state.right_hand": right_hand_state.reshape(1, -1) if args.hand == "dex3" else np.zeros((1, 6)),
                        "state.left_leg": left_leg_state.reshape(1, -1),
                        "state.right_leg": right_leg_state.reshape(1, -1),
                        "annotation.human.action.task_description": [args.goal],
                    }
                    
                    ## TODO: 这个action作为函数的return返回给LCMAgent
                    action = policy.get_action(obs)

                    pred_action_across_time = []
                    left_arm_action = action['action.left_arm']
                    right_arm_state = action['action.right_arm']
                    arm_sol_q = np.concatenate([left_arm_action, right_arm_state], axis=1)
                    
                    base_motion = action['action.base_motion']
                    left_hand_action = action['action.left_hand'] if args.hand else np.zeros((16, 6))
                    right_hand_action = action['action.right_hand'] if args.hand else np.zeros((16, 6))
                    
                    cmd_dict = {
                        'arm_action': arm_sol_q.tolist(),
                        'base_action': base_motion.tolist(),
                    }
                
                    cmd_json = json.dumps(cmd_dict)  # 转换成 list，再转成 json 字符串
                    zmq_msg = f"{topic} {cmd_json}"
                    socket.send_string(zmq_msg)
                
                if args.hand:  ### hand是从这里发出的，不知有无异步问题
                    left_q_target = np.atleast_1d(left_hand_action[count]).flatten()
                    right_q_target = np.atleast_1d(right_hand_action[count]).flatten()
                    hand_ctrl.ctrl_dual_hand(left_q_target, right_q_target)
                    count += 1
                    if count == 16:
                        count = 0
                        
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / float(args.frequency)) - time_elapsed)
                time.sleep(sleep_time)
                # print(f"main process sleep: {sleep_time}")
                    
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                

    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting program...")
    finally:     

        tv_img_shm.unlink()
        tv_img_shm.close()
        if WRIST:
            wrist_img_shm.unlink()
            wrist_img_shm.close()

        print("Finally, exiting program...")
        exit(0)
        