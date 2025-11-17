"""VR设备驱动类，基于XLeVR的VRMonitor实现

这个类提供了对VR控制器的支持，类似于SpaceMouse的接口。
使用XLeVR的VRMonitor来获取VR控制器的位置、旋转和抓取状态。

使用方法:
    1. 确保XLeVR已正确安装和配置
    2. 确保VR头显已连接并运行
    3. 在VR头显浏览器中打开XLeVR的web界面
    4. 创建VRDevice实例并开始控制

"""

import threading
import time
import numpy as np
from typing import Optional, Dict, Any

from deoxys.utils.transform_utils import rotation_matrix


class VRDevice:
    """
    VR控制器设备类，提供与SpaceMouse类似的接口
    
    Args:
        arm: 要控制的机械臂 ("left" 或 "right")
        pos_sensitivity: 位置控制灵敏度
        rot_sensitivity: 旋转控制灵敏度
        xlevr_path: XLeVR安装路径
    """
    
    def __init__(
        self, 
        arm: str = "right", 
        pos_sensitivity: float = 1.0, 
        rot_sensitivity: float = 1.0,
        xlevr_path: str = "/home/ubuntu/Downloads/XLeRobot/XLeVR"
    ):
        self.arm = arm
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.xlevr_path = xlevr_path
        
        # 导入VRMonitor
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
            from XLeVR.vr_monitor import VRMonitor
            self.VRMonitor = VRMonitor
        except ImportError as e:
            raise ImportError(
                f"无法导入VRMonitor模块: {e}\n"
                f"请确保XLeVR路径正确: {xlevr_path}\n"
                f"并且XLeVR已正确安装"
            ) from e
        
        print(f"初始化VR设备 - 控制臂: {arm}")
        
        # 6-DOF变量 (相对于上一帧的增量)
        self.dpos = np.array([0.0, 0.0, 0.0])  # 位置增量
        self.drotation = np.array([0.0, 0.0, 0.0])  # 旋转增量 (roll, pitch, yaw)
        self.raw_drotation = np.array([0.0, 0.0, 0.0])  # 原始旋转增量
        
        # 绝对位置和旋转 (用于计算增量)
        self.last_position = None
        self.last_rotation = None
        
        # 抓取状态
        self.grasp = False
        self.reset = False
        
        # 旋转矩阵 (用于坐标变换)
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        
        # VR监控器
        self.monitor = None
        self._enabled = False
        
        # 启动监听线程
        self.thread = threading.Thread(target=self._run_monitor)
        self.thread.daemon = True
        self.thread.start()
    
    def start_control(self):
        """开始VR控制"""
        print("启动VR控制...")
        self._enabled = True
    
    def stop_control(self):
        """停止VR控制"""
        print("停止VR控制...")
        self._enabled = False
    
    def _run_monitor(self):
        """运行VR监控器的主循环"""
        try:
            # 初始化VR监控器
            self.monitor = self.VRMonitor()
            if not self.monitor.initialize():
                print("❌ VR监控器初始化失败")
                return
            
            print("✅ VR监控器初始化成功")
            print("📱 请在VR头显浏览器中打开显示的HTTPS地址")
            
            # 启动监控 (这会阻塞，所以在新线程中运行)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 创建一个任务来启动监控
            async def start_monitoring_task():
                await self.monitor.start_monitoring()
            
            loop.run_until_complete(start_monitoring_task())
            
        except Exception as e:
            print(f"❌ VR监控器运行错误: {e}")
            import traceback
            print(f"错误详情: {traceback.format_exc()}")
        finally:
            # 清理资源
            if hasattr(self, 'monitor') and self.monitor:
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.monitor.stop_monitoring())
                except:
                    pass
    
    def get_controller_state(self) -> Dict[str, Any]:
        """
        获取当前控制器状态，返回与SpaceMouse兼容的格式
        
        Returns:
            Dict包含以下键:
                - dpos: 位置增量 [x, y, z]
                - rotation: 旋转矩阵
                - raw_drotation: 原始旋转增量 [roll, pitch, yaw]
                - grasp: 抓取状态
                - reset: 重置状态
        """
        if not self._enabled or self.monitor is None:
            print("VR设备未启用或监控器未初始化")
            return {
                "dpos": np.array([0.0, 0.0, 0.0]),
                "rotation": self.rotation,
                "raw_drotation": np.array([0.0, 0.0, 0.0]),
                "grasp": False,
                "reset": False,
            }
        
        # 获取当前VR目标
        goal = self.monitor.get_latest_goal_nowait(self.arm)
        
        if goal is None:
            # 没有新的VR数据，返回零增量
            print("没有新的VR数据")
            return {
                "dpos": np.array([0.0, 0.0, 0.0]),
                "rotation": self.rotation,
                "raw_drotation": np.array([0.0, 0.0, 0.0]),
                "grasp": self.grasp,
                "reset": False,
            }
        
        # 计算位置增量
        current_position = None
        if goal.target_position is not None:
            current_position = np.array(goal.target_position)
        
        # 计算旋转增量
        current_rotation = None
        if goal.wrist_roll_deg is not None and goal.wrist_flex_deg is not None:
            # 从VRMonitor获取的手腕角度（已经是相对角度）
            # wrist_roll_deg: Z轴旋转（roll）
            # wrist_flex_deg: X轴旋转（pitch）
            # wrist_yaw_deg: Y轴旋转（yaw）- 现在已正确计算
            roll = np.radians(goal.wrist_roll_deg)  # Z轴旋转
            pitch = np.radians(goal.wrist_flex_deg)  # X轴旋转
            yaw = np.radians(goal.wrist_yaw_deg) if goal.wrist_yaw_deg is not None else 0.0  # Y轴旋转
            
            # 注意：VRMonitor已经计算了相对旋转，所以这里直接使用
            current_rotation = np.array([roll, pitch, yaw])
        
        # 计算增量
        dpos = np.array([0.0, 0.0, 0.0])
        drotation = np.array([0.0, 0.0, 0.0])
        
        if current_position is not None:
            if self.last_position is not None:
                dpos = (current_position - self.last_position) * self.pos_sensitivity * 0.4
            self.last_position = current_position.copy()
        
        if current_rotation is not None:
            # VRMonitor已经计算了相对旋转，直接使用并应用灵敏度
            # 增加旋转灵敏度，使控制更敏感
            drotation = current_rotation * self.rot_sensitivity * 0.002
            # 不需要更新last_rotation，因为VRMonitor处理的是累积旋转
        
        # 更新抓取状态
        if goal.gripper_closed is not None:
            self.grasp = goal.gripper_closed
        
        # 存储原始旋转增量
        self.raw_drotation = drotation.copy()
        
        # 坐标变换：VR坐标系 -> 机器人坐标系
        # VR: [x, y, z] -> Robot: [z, x, y] (位置)
        dpos = dpos[[2, 0, 1]]
        
        # VR: [roll, pitch, yaw] -> Robot: [pitch, roll, yaw] (旋转)
        # 注意：roll是Z轴旋转，pitch是X轴旋转，yaw是Y轴旋转
        # 现在yaw已正确计算，保持正确的轴映射
        self.raw_drotation = self.raw_drotation[[1, 0, 2]]  # [pitch, roll, yaw]
        return {
            "dpos": dpos,
            "rotation": self.rotation,
            "raw_drotation": self.raw_drotation,
            "grasp": self.grasp,
            "reset": self.reset,
        }
    
    @property
    def control(self):
        """
        获取当前6-DOF控制值 (兼容SpaceMouse接口)
        
        Returns:
            np.array: 6-DOF控制值 [x, y, z, roll, pitch, yaw]
        """
        state = self.get_controller_state()
        return np.concatenate([state["dpos"], state["raw_drotation"]])
    
    @property
    def control_gripper(self):
        """
        获取抓取控制值 (兼容SpaceMouse接口)
        
        Returns:
            bool: 抓取状态
        """
        state = self.get_controller_state()
        return state["grasp"]
    
    def close(self):
        """关闭VR设备"""
        self.stop_control()
        if self.monitor:
            try:
                # 尝试优雅地停止监控器
                import asyncio
                if hasattr(self.monitor, 'stop_monitoring'):
                    # 如果监控器还在运行，尝试停止它
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            loop.create_task(self.monitor.stop_monitoring())
                    except:
                        pass
            except Exception as e:
                print(f"停止VR监控器时发生错误: {e}")
        print("VR设备已关闭")
