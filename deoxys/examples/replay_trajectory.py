"""Replay the collected trajectory using the robot arm"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller-cfg", type=str, default="osc-pose-controller.yml"
    )
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    parser.add_argument(
        "--interface-cfg", type=str, default="charmander.yml"
    )
    parser.add_argument(
        "--data-folder", type=str, required=True,
        help="Path to the folder containing recorded data (e.g., DATA/pick_test/run4)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    with open(os.path.join(args.data_folder, "config.json"), "r") as f:
        config_dict = json.load(f)
        controller_type = config_dict["controller_type"]
        controller_cfg = config_dict["controller_cfg"]

    # Load trajectory data
    ee_states = np.load(os.path.join(args.data_folder, "ee_states.npz"))["data"]
    joint_states = np.load(os.path.join(args.data_folder, "joint_states.npz"))["data"]
    
    # Initialize robot interface
    robot_interface = FrankaInterface(
        os.path.join(config_root, args.interface_cfg)
    )
    
    controller_cfg = YamlConfig(
        os.path.join(config_root, args.controller_cfg)
    ).as_easydict()

    # Reset to initial joint position
    initial_joints = joint_states[0]
    reset_joints_to(robot_interface, initial_joints)
    
    input("Press Enter to start replay...")
    
    try:
        # Replay trajectory
        for i, ee_state in enumerate(ee_states):
            # Convert ee_state (4x4 matrix in flattened form) to desired pose
            desired_pose = ee_state.reshape(4, 4)
            
            # Extract position and rotation
            position = desired_pose[:3, 3]
            rotation = desired_pose[:3, :3]
            
            # Create action vector (position + rotation + gripper)
            action = np.zeros(7)  # 3 for position, 3 for rotation, 1 for gripper
            action[:3] = position
            # Convert rotation matrix to axis-angle representation
            from scipy.spatial.transform import Rotation
            r = Rotation.from_matrix(rotation)
            action[3:6] = r.as_rotvec()
            
            # Control robot
            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            
            # Optional: add a small delay if needed
            # import time
            # time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nReplay interrupted by user")
    finally:
        robot_interface.close()

if __name__ == "__main__":
    main()
