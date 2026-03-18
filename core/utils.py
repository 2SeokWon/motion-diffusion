# utils.py

import torch
import numpy as np
import os
import math
from pyglm import glm
from bvh_viewer.BVH_Parser import Motion, Joint, get_preorder_joint_list
from .kinematics import mat_to_euler_yxz, quat_to_euler

template_path = os.path.join(os.path.dirname(__file__), "../data/raw/Drunk_TR1.bvh")

def write_bvh(root: Joint, motion_obj: Motion, output_path: str):
    print(f"Writing BVH file to: {output_path}")
    
    with open(template_path, 'r') as f: lines = f.readlines()
    header_end_index = -1
    for i, line in enumerate(lines):
        if "MOTION" in line.upper(): header_end_index = i; break
    header_lines = lines[:header_end_index]
    
    with open(output_path, 'w') as f:
        f.writelines(header_lines)
        f.write("MOTION\n")
        f.write(f"Frames: {motion_obj.frame_len}\n")
        f.write(f"Frame Time: {motion_obj.frame_time}\n")
        i = 0
        for frame in motion_obj.quaternion_frame:
            global_hip_matrix = np.array(frame.joint_global_transforms[root.name])
            hip_pos = global_hip_matrix[:3,3]
            hip_rot = global_hip_matrix[:3,:3]
            hip_euler_rad = mat_to_euler_yxz(hip_rot)
            hip_euler_deg = [math.degrees(a) for a in hip_euler_rad]
            line = f"{hip_pos[0]:.6f} {hip_pos[1]:.6f} {hip_pos[2]:.6f} {hip_euler_deg[0]:.6f} {hip_euler_deg[1]:.6f} {hip_euler_deg[2]:.6f} "
            for joint_name in get_preorder_joint_list(root):
                if joint_name.name == root.name:
                    continue
                joint_local_rotation = frame.joint_rotations[joint_name.name]
                euler_rad = quat_to_euler(joint_local_rotation)
                euler = [math.degrees(a) for a in euler_rad]
                line += f"{euler[0]:.6f} {euler[1]:.6f} {euler[2]:.6f} "

            f.write(line.strip() + "\n")
            i += 1
    print("BVH file writing complete.")