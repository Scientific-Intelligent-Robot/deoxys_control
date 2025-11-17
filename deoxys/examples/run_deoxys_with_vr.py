import argparse
import time
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import VRDevice
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()


def main():
    parser = argparse.ArgumentParser(description="使用VR控制器控制Deoxys机器人")
    parser.add_argument("--interface-cfg", type=str, default="config/charmander.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    parser.add_argument("--arm", type=str, default="right", choices=["left", "right"], 
                       help="要控制的机械臂")

    args = parser.parse_args()

    # 创建VR设备
    device = VRDevice(arm=args.arm)
    
    # 等待VR监控器初始化
    print("等待VR监控器初始化...")
    time.sleep(3)  # 给VR监控器一些时间初始化
    
    device.start_control()

    # 创建机器人接口
    robot_interface = FrankaInterface(args.interface_cfg, use_visualizer=False)

    controller_type = args.controller_type
    controller_cfg = get_default_controller_config(controller_type=controller_type)

    robot_interface._state_buffer = []

    print(f"开始VR控制 - 控制臂: {args.arm}, 控制器类型: {controller_type}")
    print("在VR头显中移动控制器来控制机器人...")
    print("按Ctrl+C停止控制")

    try:
        while(True):
            start_time = time.time_ns()

            # 获取VR输入并转换为机器人动作
            action, grasp = input2action(
                device=device,
                controller_type=controller_type,
            )

            print(action)
            
            # 发送控制命令到机器人
            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            
            end_time = time.time_ns()
            logger.debug(f"时间间隔: {((end_time - start_time) / (10**9))}")
            
            # 短暂休眠以避免过度占用CPU
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n收到停止信号，正在安全停止...")
    except Exception as e:
        print(f"控制过程中发生错误: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
    finally:
        # 发送停止命令
        # robot_interface.control(
        #     controller_type=controller_type,
        #     action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [1.0],
        #     controller_cfg=controller_cfg,
        #     termination=True,
        # )

        # robot_interface.close()
        device.close()

        # 检查是否有状态帧丢失
        # print("检查状态帧...")
        # for (state, next_state) in zip(
        #     robot_interface._state_buffer[:-1], robot_interface._state_buffer[1:]
        # ):
        #     if (next_state.frame - state.frame) > 1:
        #         print(f"发现丢失的帧: {state.frame} -> {next_state.frame}")


if __name__ == "__main__":
    main()