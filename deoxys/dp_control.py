import socket
import pickle
import time
import numpy as np
import os
import copy
import threading

import cv2
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

class CameraReaderThread(threading.Thread):
    """独立的相机读取线程，持续读取最新图像"""
    def __init__(self, multi_camera, camera_ids):
        super().__init__(daemon=True)
        self.multi_camera = multi_camera
        self.camera_ids = camera_ids
        self.latest_images = {}  # 存储最新的图像
        self.lock = threading.Lock()  # 保护共享数据
        self.stop_event = threading.Event()  # 停止信号
        self.frame_count = 0  # 相机线程的帧计数
        
    def run(self):
        """持续读取相机图像"""
        print("[CAMERA_THREAD] 相机读取线程已启动")
        while not self.stop_event.is_set():
            try:
                images = self.multi_camera.get_frame()
                camera_images = {}
                for camera_id in self.camera_ids:
                    rgb, depth = images[camera_id]
                    camera_images[camera_id] = copy.deepcopy(rgb)
                
                # 使用锁更新最新图像
                with self.lock:
                    self.latest_images = camera_images
                    self.frame_count += 1
                    
            except Exception as e:
                print(f"[CAMERA_THREAD] 读取图像出错: {e}")
                time.sleep(0.01)
                
    def get_latest_images(self):
        """获取最新的图像副本（线程安全）"""
        with self.lock:
            return copy.deepcopy(self.latest_images), self.frame_count
    
    def stop(self):
        """停止线程"""
        self.stop_event.set()
        print("[CAMERA_THREAD] 相机读取线程已停止")

def main():
    # 初始化机械臂和相机
    robot_interface = FrankaInterface(os.path.join(config_root, "charmander.yml"))
    multi_camera = MultiCamera(serial_numbers=["CP2R553000YK", "CP2R553000FZ", "CP026530002N"], camera_width=640, camera_height=480, camera_fps=30)
    
    # 标记是否可以使用可视化（首次尝试后确定）
    enable_visualization = True

    camera_ids = list(multi_camera.cameras.keys())
    controller_cfg = YamlConfig(os.path.join(config_root, "osc-pose-controller.yml")).as_easydict()
    controller_type = "OSC_POSE"

    reset_joint_positions = [0.076, -1.036, -0.054, -2.384, -0.0045, 1.382, 0.785]
    reset_joints_to(robot_interface, reset_joint_positions)

    # 启动独立的相机读取线程
    camera_thread = CameraReaderThread(multi_camera, camera_ids)
    camera_thread.start()
    print("[CLIENT] 相机读取线程已启动，等待图像...")
    
    # 等待相机线程获取第一帧
    while True:
        camera_images, _ = camera_thread.get_latest_images()
        if len(camera_images) > 0:
            print("[CLIENT] 已获取到初始图像")
            break
        time.sleep(0.01)

    # 建立socket连接
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    print("[CLIENT] Connected to server.")

    # 主线程的发送帧计数器
    send_count = 0

    try:
        while True:
            send_count += 1
            # 获取机械臂状态
            if len(robot_interface._state_buffer) == 0:
                continue
            last_state = robot_interface._state_buffer[-1]
            last_gripper_state = robot_interface._gripper_state_buffer[-1]
            
            if last_gripper_state.width is None:
                continue
            state_dict = {
                "ee_states": np.array(last_state.O_T_EE),
                "joint_states": np.array(last_state.q),
                "joint_vel": np.array(last_state.dq),
                "gripper_states": np.array(last_gripper_state.width),
            }

            # 从相机线程获取最新图像（线程安全）
            camera_images, camera_frame_count = camera_thread.get_latest_images()
            
            # 可视化所有相机图像（拼接显示）
            if enable_visualization:
                try:
                    images_to_display = []
                    for cam_id in camera_ids:
                        img = camera_images[cam_id].copy()
                        # 在每个相机图像上添加帧数显示
                        cv2.putText(img, f"Sent: {send_count}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(img, f"Cam: {camera_frame_count}", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        images_to_display.append(img)
                    
                    combined_image = np.hstack(images_to_display)  # 水平拼接
                    cv2.imshow("Camera Views", combined_image)
                    cv2.waitKey(1)  # 非阻塞刷新
                    
                    # 打印帧数和时间戳
                    print(f"[CLIENT] Send {send_count} | Camera {camera_frame_count} | Time: {time.time():.3f}")
                except Exception as e:
                    print(f"[WARNING] 无法显示图像窗口: {e}")
                    print("[WARNING] 禁用可视化功能，继续运行...")
                    enable_visualization = False
        
            obs = {
                "ee_states": state_dict["ee_states"],
                "joint_states": state_dict["joint_states"],
                "gripper_states": state_dict["gripper_states"],
            }
            for camera_id in camera_ids:
                obs[f"{camera_id}"] = camera_images[camera_id]
            # 发送观测
            s.sendall(pickle.dumps(obs) + b"<END>")
            print("[CLIENT] Sent observation.")

            # 接收推理结果
            data = b""
            while True:
                packet = s.recv(4096)
                if not packet:
                    print("[CLIENT] Server disconnected.")
                    camera_thread.stop()
                    return
                data += packet
                if data.endswith(b"<END>"):
                    data = data[:-5]
                    break

            action = pickle.loads(data)[0]
            print(f"[CLIENT] Received action: {action}")
            # 执行动作
            for i in range(len(action)):
                robot_interface.control(
                    controller_type=controller_type,
                    action=action[i],
                    controller_cfg=controller_cfg,
                )

    
    except KeyboardInterrupt:
        print("\n[CLIENT] 检测到中断信号，正在停止...")
        camera_thread.stop()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[CLIENT] 发生错误: {e}")
        camera_thread.stop()
        cv2.destroyAllWindows()
        raise

if __name__ == "__main__":
    main()
