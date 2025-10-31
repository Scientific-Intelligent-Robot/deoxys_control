"""Teleoperating robot arm with a SpaceMouse to collect demonstration data"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger

from deoxys.camera.orbbec_camera import MultiCamera

logger = get_deoxys_example_logger()
# reset_joint_positions = [0, 0, 0, -2, 0, 2, 0.79]
# reset_joint_positions = [0.14864346745997561, -0.22840032580307748, -0.12616538019975027, -2.0411589933489056, 0.010689137402532738, 1.876730053451326, 0.9139136976603831]

reset_joint_positions = [
    0.0760389047913384,
    -1.0362613022620384,
    -0.054254247684777324,
    -2.383951857286591,
    -0.004505598470154735,
    1.3820559157131187,
    0.784935455988679,
]  # normal

# reset_joint_positions = [
#     0.07204696,
#     0.4065654,
#     0.131453,
#     -2.14534333,
#     -0.00582897,
#     2.6131106,
#     1.02968312,
# ]  # push T


# reset_joint_positions =[0.13162655, 0.21720941, 0.08970448, -1.71676526, 0.05230191, 1.81093706, 0.77421045] # cabinet
# reset_joint_positions = [ 0.14818035, -0.08560756,  0.16768701, -1.6259418 ,  0.05298611, 1.60141484,  0.80027921]
# reset_joint_positions = [ 0.46252498, -0.55639266,  0.22756037, -2.21484251,  0.13145469, 1.6528019 ,  0.8643308 ] # drawer
# reset_joint_positions = [-0.02215226023735707, -0.8626876594993124, 0.06338327341748957, -2.4240048837156674, -0.6669373935781613, 3.094094994107882, 1.5016620106963892]
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vendor_id",
        type=int,
        default=9583,
    )
    parser.add_argument(
        "--product_id",
        type=int,
        default=50746,  # 50741 for flexiv 50734 for franka
    )
    parser.add_argument(
        "--controller-cfg", type=str, default="osc-pose-controller.yml"
    )
    parser.add_argument("--controller-type", type=str, default="OSC_POSITION")
    parser.add_argument(
        "--interface-cfg", type=str, default="charmander.yml"
    )
    parser.add_argument(
        "--folder", type=Path, default="./DATA"
    )
    parser.add_argument("--task_name", type=Path, default="test_maple")
    parser.add_argument("--save_depth", type=bool, default=False)
    parser.add_argument("--camera_width", type=int, default=640, help="Camera resolution width")
    parser.add_argument("--camera_height", type=int, default=480, help="Camera resolution height")
    parser.add_argument("--camera_fps", type=int, default=30, help="Camera frame rate")
    parser.add_argument("--auto_white_balance", type=bool, default=False, help="Enable auto white balance")
    parser.add_argument("--white_balance_temp", type=int, default=4500, help="White balance temperature in Kelvin (ignored if auto_white_balance is True)")

    args = parser.parse_args()
    args.folder = args.folder / args.task_name
    return args


def main():
    args = parse_args()
    print(args)
    args.folder.mkdir(parents=True, exist_ok=True)

    experiment_id = 0

    logger.info(f"Saving to {args.folder}")

    # Create a folder that saves the demonstration raw states.
    for path in args.folder.glob("run*"):
        if not path.is_dir():
            continue
        try:
            folder_id = int(str(path).split("run")[-1])
            if folder_id > experiment_id:
                experiment_id = folder_id
        except BaseException:
            pass
    experiment_id += 1
    folder = str(args.folder / f"run{experiment_id}")

    device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
    device.start_control()

    print("config_root", config_root)

    # Franka Interface
    robot_interface = FrankaInterface(
        os.path.join(config_root, args.interface_cfg)
    )
    multi_camera = MultiCamera(
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        auto_white_balance=args.auto_white_balance,
        white_balance_temp=args.white_balance_temp
    )
    camera_ids = list(multi_camera.cameras.keys())

    controller_cfg = YamlConfig(
        os.path.join(config_root, args.controller_cfg)
    ).as_easydict()

    controller_type = args.controller_type

    data = {
        "action": [],
        "ee_states": [],
        "joint_states": [],
        "gripper_states": [],
        "joint_vel": [],
    }
    for camera_id in camera_ids:
        data[f"camera_{camera_id}"] = []
        if args.save_depth:
            data[f"camera_{camera_id}_depth"] = []
    i = 0
    start = False

    previous_state_dict = None

    reset_joints_to(robot_interface, reset_joint_positions)
    # input(f"Press Enter to start episode...")
    # time.sleep(2)

    while True:
        if len(robot_interface._state_buffer) == 0:
            # import ipdb
            # ipdb.set_trace()
            continue
        last_state = robot_interface._state_buffer[-1]
        last_gripper_state = robot_interface._gripper_state_buffer[-1]

        state_dict = {
            "ee_states": np.array(last_state.O_T_EE),
            "joint_states": np.array(last_state.q),
            "joint_vel": np.array(last_state.dq),
            "gripper_states": np.array(last_gripper_state.width),
        }

        print("the ee_states is:", state_dict["ee_states"])

        # Get img info
        images = multi_camera.get_frame()
        rgbs = []
        depths = []

        for camera_id in camera_ids:
            rgb, depth = images[camera_id]

            # resize_shape = (448, 448)
            # rgb = cv2.resize(rgb, resize_shape)
            # depth = cv2.resize(depth, resize_shape)

            rgbs.append(rgb)
            # depths.append(depth)
            if start:
                data[f"camera_{camera_id}"].append(rgb.copy())
                if args.save_depth:
                    data[f"camera_{camera_id}_depth"].append(depth.copy())
        
            # depth = cv2.resize(depth, (224, 224))
            # data[f"camera_{camera_id}"].append(np.concatenate([rgb, depth[:, :, np.newaxis]], axis=-1))
        
        # cv2.imshow("rgb", np.hstack(rgbs.copy()))
        cv2.waitKey(1)
        end_time = time.time_ns()

        # print(f"Time profile: {(end_time - start_time) / 10 ** 9}")

        i += 1
        start_time = time.time_ns()
        action, grasp = input2action(
            device=device,
            controller_type=controller_type,
        )
        if action is None:
            break

        # set unused orientation dims to 0
        if controller_type == "OSC_YAW":
            action[3:5] = 0.0
        elif controller_type == "OSC_POSITION":
            action[3:6] = 0.0
        # action[-1] = 1.0
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )

        if np.linalg.norm(action[:-1]) < 1e-3 and not start:
            continue

        start = True
        if start:
            for proprio_key in state_dict.keys():
                data[proprio_key].append(state_dict[proprio_key])
            data["action"].append(action)

        # time.sleep(0.28)

    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/config.json", "w") as f:
        config_dict = {
            "controller_cfg": dict(controller_cfg),
            "controller_type": controller_type,
        }
        json.dump(config_dict, f)
        np.savez(f"{folder}/action", data=np.array(data["action"]))
        np.savez(f"{folder}/ee_states", data=np.array(data["ee_states"]))
        np.savez(f"{folder}/joint_states", data=np.array(data["joint_states"]))
        np.savez(f"{folder}/joint_vel", data=np.array(data["joint_vel"]))
        np.savez(
            f"{folder}/gripper_states",
            data=np.array(data["gripper_states"]),
        )

    for camera_id in camera_ids:
        # Save video using OpenCV
        frames = data[f"camera_{camera_id}"]
        if frames:
            height, width = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                f"{folder}/camera_{camera_id}.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'),
                20,  # fps
                (width, height)
            )
            for frame in frames:
                if frame is not None:
                    writer.write(frame)
            writer.release()
        
        # Save raw image data
        np.savez(
            f"{folder}/camera_{camera_id}",
            data=np.array(data[f"camera_{camera_id}"]),
        )
        if args.save_depth:
            np.savez(
                f"{folder}/camera_{camera_id}_depth",
                data=np.array(data[f"camera_{camera_id}_depth"]),
            )

    # Close all devices
    robot_interface.close()
    multi_camera.close()
    print("Total length of the trajectory: ", len(data["action"]))
    valid_input = False
    while not valid_input:
        try:
            save = input("Save or not? (enter 0 or 1)")
            save = bool(int(save))
            valid_input = True
        except:
            pass
    print("Saving: ", save)
    if not save:
        import shutil
        shutil.rmtree(f"{folder}")
    
    reproduce = bool(int(input("reproduce? (enter 0 or 1)")))
    if reproduce:
        actions = np.load(f"{folder}/action.npz")["data"]
        reset_joints_to(robot_interface, reset_joint_positions)
        for action in actions:
            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            time.sleep(0.05)


if __name__ == "__main__":
    main()
