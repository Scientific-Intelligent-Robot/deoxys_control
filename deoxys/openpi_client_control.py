import os
import sys
import time
import logging
from PIL import Image
import numpy as np
from pathlib import Path
import cv2
import einops

# 添加 openpi-client 到路径
sys.path.insert(0, str(Path.home() / "kl_workspace" / "openpi-client" / "src"))

from openpi_client.websocket_client_policy import WebsocketClientPolicy

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.camera.orbbec_camera import MultiCamera


# ============ 配置参数 ============ #
# WebSocket 服务器配置
WEBSOCKET_HOST = "localhost"
WEBSOCKET_PORT = 8000

# 相机配置
CAMERA_SERIALS = ["CP2R553000YK", "CP2R553000FZ", "CP026530002N"]
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# 机械臂初始位置
RESET_JOINT_POSITIONS = [0.076, -1.036, -0.054, -2.384, -0.0045, 1.382, 0.785]

def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format, center crop and resize to 224x224."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    
    # 居中裁切成正方形 (480x480)
    h, w = image.shape[:2]
    crop_size = min(h, w)
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    image = image[start_h:start_h + crop_size, start_w:start_w + crop_size]
    
    # 缩放到 224x224
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    return image

def prepare_observation(state_dict, camera_images, camera_ids):
    """
    准备符合 OpenPI 格式的观测数据
    
    Args:
        state_dict: 机械臂状态字典
        camera_images: 相机图像字典
        camera_ids: 相机ID列表
    
    Returns:
        obs: 格式化的观测字典
    """
    obs = {}
    
    obs["observation/state"] = np.concatenate([state_dict["joint_states"].flatten(), state_dict["gripper_states"].flatten()], axis=0).astype(np.float32)
    # 添加相机图像（转换为 uint8 以减小传输大小）
    obs["observation/image"] = _parse_image(camera_images["CP2R553000FZ"])
    obs["observation/wrist_image"] = _parse_image(camera_images["CP026530002N"])
    obs["observation/right_wrist_image"] = _parse_image(camera_images["CP2R553000YK"])
    # obs["prompt"] = "pick up the organe pumpki"
    return obs


def execute_action(robot_interface, action, controller_type, controller_cfg):
    """
    执行单步动作
    
    Args:
        robot_interface: 机械臂接口
        action: 动作数组
        controller_type: 控制器类型
        controller_cfg: 控制器配置
    """
    robot_interface.control(
        controller_type=controller_type,
        action=action,
        controller_cfg=controller_cfg,
    )


def main():
    # 配置日志
    logging.basicConfig(
        level=print,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    print("初始化机械臂和相机...")
    
    # 初始化机械臂
    robot_interface = FrankaInterface(
        os.path.join(config_root, "charmander.yml")
    )
    
    # 初始化相机
    multi_camera = MultiCamera(
        serial_numbers=CAMERA_SERIALS,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT,
        camera_fps=CAMERA_FPS
    )
    camera_ids = list(multi_camera.cameras.keys())
    
    # 控制器配置
    controller_cfg = YamlConfig(
        os.path.join(config_root, "osc-pose-controller.yml")
    ).as_easydict()
    controller_type = "OSC_POSE"
    
    # 重置机械臂到初始位置
    print("重置机械臂到初始位置...")
    reset_joints_to(robot_interface, RESET_JOINT_POSITIONS)
    
    # 连接到 OpenPI 服务器
    print(f"连接到 OpenPI 服务器 ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}...")
    policy = WebsocketClientPolicy(host=WEBSOCKET_HOST, port=WEBSOCKET_PORT)
    
    # 获取服务器元数据
    server_metadata = policy.get_server_metadata()
    print(f"服务器元数据: {server_metadata}")
    
    print("开始控制循环...")
    
    step_count = 0

    while True:
        
        # 获取机械臂状态
        if len(robot_interface._state_buffer) == 0:
            continue
        
        last_state = robot_interface._state_buffer[-1]
        last_gripper_state = robot_interface._gripper_state_buffer[-1]
        
        # 检查 gripper 状态是否有效
        if last_gripper_state.width is None:
            continue
        
        # 构建状态字典
        state_dict = {
            "ee_states": np.array(last_state.O_T_EE),
            "joint_states": np.array(last_state.q),
            "joint_vel": np.array(last_state.dq),
            "gripper_states": np.array(last_gripper_state.width),
        }
        
        # 获取相机图像
        images = multi_camera.get_frame()
        camera_images = {}
        for camera_id in camera_ids:
            rgb, depth = images[camera_id]
            camera_images[camera_id] = rgb
                        
        obs = prepare_observation(state_dict, camera_images, camera_ids)
        
        # 推理获取动作
        action_result = policy.infer(obs)
        actions = action_result.get("actions", None)
        
        if actions is None:
            print("未收到有效动作，跳过本次循环")
            continue
        
        print(actions[:10])
        # 执行动作
        for action in actions[:10]:
            execute_action(robot_interface, action, controller_type, controller_cfg)
        
        step_count += 1

        print(f"已执行 {step_count} 步")

if __name__ == "__main__":
    main()

