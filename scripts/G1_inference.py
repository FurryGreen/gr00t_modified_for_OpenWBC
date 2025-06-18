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
from inference_deploys.robot_control.filter import ActionFilter, MovingAverageFilter, ExponentialMovingAverageFilter, SavitzkyGolayFilter

import pickle
import socket


infer_flag = False
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

import numpy as np
import matplotlib.pyplot as plt

def plot_predicted_actions(pred_action_across_time, state_across_time, modality_keys, action_horizon=16, traj_id=0, save_path=None):
    """
    绘制预测的动作轨迹。

    Args:
        pred_action_across_time (np.ndarray): 形状为 (T, D) 的预测动作轨迹，T为时间步数，D为动作维度。
        modality_keys (list of str): 表示动作模态的键列表。
        action_horizon (int): 每隔多少步标注一次inference点。
        traj_id (int): 当前轨迹的编号（用于图标题）。
        save_path (str or None): 如果提供，将保存图像到此路径，否则只展示图像。
    """
    pred_action_across_time = np.array(pred_action_across_time)
    steps, action_dim = pred_action_across_time.shape

    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(8, 4 * action_dim))
    fig.suptitle(f"Trajectory {traj_id} - Modalities: {', '.join(modality_keys)}", fontsize=16, color="blue")

    if action_dim == 1:
        axes = [axes]  # 保证可迭代性

    for i, ax in enumerate(axes):
        ax.plot(pred_action_across_time[:, i], label="pred action")
        count = 0
        for j in range(0, steps, action_horizon):
            label = "inference point" if j == 0 else None
            ax.plot(j, pred_action_across_time[j, i], "ro", label=label)
        #### 加入state_across_time绘制代码，注意state只在action的inference point处绘制
            ax.plot(j, state_across_time[count][i], "gx", label="state" if (j == 0) else None)
            count += 1
        ######
        ax.set_title(f"Predicted Action Dimension {i}")
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留出 suptitle 空间

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def smooth_actions(action_seq: np.ndarray, filter: ActionFilter) -> np.ndarray:
    """
    对 action_seq 的最后16个动作进行平滑处理。
    
    参数:
    - action_seq: (T, D) 的动作序列
    - filter: 实现了 apply() 方法的滤波器对象
    
    返回:
    - (16, D) 的平滑动作序列
    """
    assert action_seq.shape[0] >= 16, "动作序列不足16步"
    return filter.apply(action_seq)[-16:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type = str, default = './utils/data', help = 'path to save data')
    parser.add_argument('--frequency', type = int, default = 30.0, help = 'save data\'s frequency')

    parser.add_argument('--vis', action = 'store_true', help = 'Save data or not')
    parser.add_argument('--no-record', dest = 'record', action = 'store_false', help = 'Do not save data')
    parser.set_defaults(record = False)

    parser.add_argument('--arm', type=str, default = 'G1_29', choices=['G1_29', 'G1_23', 'H1_2', 'H1'], help='Select arm controller')
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
    parser.add_argument('--filt', action = 'store_true', help = 'add filter')
    

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
    arm_keys = ['left_arm', 'right_arm', 'left_hand', 'right_hand']#modality_keys
    arm_keys = ['base_motion', 'right_hand', 'right_arm']
    #arm_keys = ['left_hand', 'right_hand']
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
        img_client = ImageClient(image_show = False,tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name)

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
            num = 0
            recover_base_motion = None
            if args.vis:
                pred_action_across_time = []
                state_across_time = []
            if args.filt:
                act_filter = ExponentialMovingAverageFilter(0.3, length =48)  # 0.9 is the smoothing factor
                action_seq = {}
                for key in modality_keys:
                    action_seq[f'action.{key}'] = None

            init_state = {'left_arm': np.array([-0.28927523,  0.03668371,  0.4204905 ,  0.25827205, -0.04727777,
                    0.03626331, -0.87588775]), 
                    'right_arm': np.array([-0.24665932, -0.15940218, -0.26253843,  0.12521118, -0.00215716,
                    0.02495787,  0.68305868]), 
                    'left_hand': np.array([-0.8472293 ,  0.7052123 ,  0.0249148 , -0.03309153, -0.01703873,
                -0.08243784, -0.01271194]), 
                'right_hand': np.array([-0.79104125, -0.81918585, -0.01530439,  0.02775995,  0.0234479 ,
                    0.02790572,  0.03036809]), 
                    'left_leg': np.array([-0.28927523,  0.03668371,  0.4204905 ,  0.25827205, -0.04727777,
                    0.03626331]), 
                    'right_leg': np.array([-0.87588775, -0.24665932, -0.15940218, -0.26253843,  0.12521118,
                -0.00215716]),
                    'base_motion': np.array([0.0, 0.0, 0.0, 0.74])}
            
            #### 这里开始可以包装为函数：model_infer_mode
            if args.hand:  ### hand是从这里发出的，不知有无异步问题
                left_q_target = np.zeros(7)
                right_q_target = np.zeros(7)
                #print('hand out:', left_q_target, right_q_target)
                hand_ctrl.ctrl_dual_hand(left_q_target, right_q_target)
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
                            left_hand_state = np.array(dual_hand_state_array[:7])
                            right_hand_state = np.array(dual_hand_state_array[-7:])

                    elif args.hand == "gripper":
                        with dual_gripper_data_lock:
                            left_hand_state = [dual_gripper_state_array[1]]
                            right_hand_state = [dual_gripper_state_array[0]]
                    # print('wyx debug:', left_hand_state)
                    H,W,C = tv_img_array.shape
                    tv_resized_image = cv2.resize(tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
                    cv2.imwrite('Image_Debug.png', tv_resized_image)
                    #cv2.imshow('Image_Debug', tv_resized_image)
                    # cv2.waitKey(1)
                    #img_obs = cv2.cvtColor(copy.deepcopy(tv_img_array), cv2.COLOR_BGR2RGB)
                    
                    img_obs = copy.deepcopy(tv_img_array)
                    if num == 0:
                        # 用init state赋值
                        action = {}
                        for key in ['left_arm', 'right_arm', 'left_hand', 'right_hand', 'base_motion']:
                            action[f'action.{key}'] = np.tile(init_state[key], (16, 1))  # 16 steps
                        print('initialize state')
                    else:
                        #    img_obs = np.zeros_like(img_obs)
                        obs = {
                            "video.ego_view": img_obs.reshape(1, H,W,C), ### important: COPY!!!
                            "state.left_arm": left_arm_state.reshape(1, -1),
                            "state.right_arm": right_arm_state.reshape(1, -1),
                            "state.left_hand": left_hand_state.reshape(1, -1) if args.hand == "dex3" else np.zeros((1, 6)),
                            "state.right_hand": right_hand_state.reshape(1, -1) if args.hand == "dex3" else np.zeros((1, 6)),
                            "state.left_leg": left_leg_state.reshape(1, -1),
                            "state.right_leg": right_leg_state.reshape(1, -1),
                            "annotation.human.action.task_description": [args.goal],
                        }
                        
                        start_time_p = time.time()
                        action = policy.get_action(obs)
                        interval = time.time() - start_time_p
                        print(f"Policy inference time: {interval:.4f} seconds")

                        
                    if args.filt:
                        smooth_action = {}
                        for key in modality_keys:
                            if action_seq[f'action.{key}'] is None:
                                action_seq[f'action.{key}'] = np.array(action[f'action.{key}'])
                            else:
                                action_seq[f'action.{key}'] = np.concatenate([action_seq[f'action.{key}'], np.array(action[f'action.{key}'])], axis=0)
                            smooth_action[f'action.{key}'] = smooth_actions(action_seq[f'action.{key}'], act_filter)
                        action = smooth_action
                        
                    if args.vis:
                        for j in range(16):
                            # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                            # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                            concat_pred_action = np.concatenate(
                                [np.atleast_1d(action[f"action.{key}"][j]) for key in arm_keys],
                                axis=0,
                            )
                            pred_action_across_time.append(concat_pred_action)

                        stats= []
                        for key in arm_keys:
                            try:
                                stats.append(np.atleast_1d(obs[f"state.{key}"][0]))
                            except:
                                stats.append(np.zeros_like(action[f"action.{key}"][0]))
                        concat_state = np.concatenate(stats, axis=0)
                        state_across_time.append(concat_state)
                    
                        
                    left_arm_action = action['action.left_arm']
                    right_arm_state = action['action.right_arm']
                    arm_sol_q = np.concatenate([left_arm_action, right_arm_state], axis=1)
                    
                    base_motion = action['action.base_motion'] if recover_base_motion is None else recover_base_motion
                    # if np.any(base_motion[:, 3] > 0.6) and num > 400:
                    #     print("starting recover")
                    #     start_value = base_motion[15, 3]
                    #     end_value = 0.74
                    #     # 生成线性变化的第4列数据
                    #     new_col4 = np.linspace(start_value, end_value, 16)
                    #     # 创建新的 base_motion，前三列保持不变或初始化为0
                    #     recover_base_motion = np.copy(base_motion)
                    #     recover_base_motion[:, 3] = new_col4


                    left_hand_action = action['action.left_hand'] if args.hand else np.zeros((16, 6))
                    right_hand_action = action['action.right_hand'] if args.hand else np.zeros((16, 6))
                    
                    cmd_dict = {
                        'arm_action': arm_sol_q.tolist(),
                        'base_action': base_motion.tolist(),
                    }
                
                    cmd_json = json.dumps(cmd_dict)  # 转换成 list，再转成 json 字符串
                    zmq_msg = f"{topic} {cmd_json}"
                    socket.send_string(zmq_msg)

                num += 1

                if args.hand:  ### hand是从这里发出的，不知有无异步问题
                    if num < 60:
                        left_q_target = np.zeros(7)
                        right_q_target = np.zeros(7)
                        #print('hand out:', left_q_target, right_q_target)
                        hand_ctrl.ctrl_dual_hand(left_q_target, right_q_target)
                    else:
                        if current_flag:
                            count = 0
                        left_q_target = np.atleast_1d(left_hand_action[count]).flatten()
                        right_q_target = np.atleast_1d(right_hand_action[count]).flatten()
                        #print('hand out:', left_q_target, right_q_target)
                        hand_ctrl.ctrl_dual_hand(left_q_target, right_q_target)
                        count += 1
                        if count > 15:
                            count = 15
                        
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

        if args.vis:
            pred_action_across_time = np.array(pred_action_across_time)[:1500]
            state_across_time = np.array(state_across_time)[:1500]
            print('debug:', pred_action_across_time.shape)
            plot_predicted_actions(pred_action_across_time, state_across_time, arm_keys, action_horizon=args.action_horizon, traj_id=0, save_path=os.path.join('predicted_actions.png'))
            
        print("Finally, exiting program...")
        exit(0)
        