import socket
import pickle
import time
import numpy as np
import cv2
import os
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.camera.orbbec_camera import MultiCamera

# 推理端启动监听再运行：
# python client_control.py
# ============ Socket配置 ============ #
HOST = "127.0.0.1"
PORT = 50007

def main():
    # 初始化机械臂和相机
    robot_interface = FrankaInterface(os.path.join(config_root, "charmander.yml"))
    multi_camera = MultiCamera(serial_numbers=["CP026530002N"], camera_width=640, camera_height=480, camera_fps=30)

    camera_ids = list(multi_camera.cameras.keys())
    controller_cfg = YamlConfig(os.path.join(config_root, "osc-pose-controller.yml")).as_easydict()
    controller_type = "OSC_POSE"

    reset_joint_positions = [0.076, -1.036, -0.054, -2.384, -0.0045, 1.382, 0.785]
    reset_joints_to(robot_interface, reset_joint_positions)

    # 建立socket连接
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    print("[CLIENT] Connected to server.")

    while True:
        # 获取机械臂状态
        if len(robot_interface._state_buffer) == 0:
            continue
        last_state = robot_interface._state_buffer[-1]
        last_gripper_state = robot_interface._gripper_state_buffer[-1]
        
        if last_gripper_state.width is None:
            continue
        print(last_gripper_state.width)
        state_dict = {
            "ee_states": np.array(last_state.O_T_EE),
            "joint_states": np.array(last_state.q),
            "joint_vel": np.array(last_state.dq),
            "gripper_states": np.array(last_gripper_state.width),
        }

        images = multi_camera.get_frame()
        rgbs = []
        for camera_id in camera_ids:
            rgb, depth = images[camera_id]
            rgbs.append(rgb)
        obs = {
            "ee_states": state_dict["ee_states"],
            "joint_states": state_dict["joint_states"],
            "gripper_states": state_dict["gripper_states"],
            "camera": np.stack(rgbs, axis=0)
        }

        # 发送观测
        s.sendall(pickle.dumps(obs) + b"<END>")
        print("[CLIENT] Sent observation.")

        # 接收推理结果
        data = b""
        while True:
            packet = s.recv(4096)
            if not packet:
                print("[CLIENT] Server disconnected.")
                return
            data += packet
            if data.endswith(b"<END>"):
                data = data[:-5]
                break

        action = pickle.loads(data)
        print(f"[CLIENT] Received action: {action}")
        # 执行动作
        for i in range(action.shape[0]):
            robot_interface.control(
                controller_type=controller_type,
                action=action[i],
                controller_cfg=controller_cfg,
            )

        time.sleep(0.05)

if __name__ == "__main__":
    main()
