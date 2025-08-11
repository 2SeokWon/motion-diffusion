#bvh_parser.py

import numpy as np
from pyglm import glm
from Transforms import translation_matrix, extract_vroot_transform

def mat4_close(a, b, eps=1e-4):
    for i in range(3):
        for j in range(3):
            if abs(a[i][j] - b[i][j]) > eps:
                print(f"Mismatch at ({i},{j}): {a[i][j]} vs {b[i][j]}")
                return False
    return True

class Joint:
    """
    관절을 정의하는 Joint Class
    """
    def __init__(self, name):
        self.name = name
        self.channels = []
        self.children = []
        self.offset = [0, 0, 0]
        self.kinematics = glm.mat4(1.0)
        self.parent = None

class Motion:
    def __init__(self, frames, frame_time, frame_len):
        self.frames = frames #모든 frame이 list로 들어가있음.[[],[],...,[]]
        self.frame_time = frame_time
        self.frame_len = frame_len
        self.quaternion_frame = []

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.frames):
            raise IndexError("Frame index out of range")
        return self.frames[idx]

    def list_to_quaternion(self, root):
        self.quaternion_frame = []
        for frame in self.frames:
            motion_frame = MotionFrame()
            process_joint(root, frame, motion_frame, 0)
            self.quaternion_frame.append(motion_frame)

    def save_virtual_root_info(self, root):
        for frame in self.quaternion_frame:
            root_global_pos = frame.joint_positions.get(root.name, glm.vec3(0, 0, 0))
            root_global_rot = frame.joint_rotations.get(root.name, glm.quat(1, 0, 0, 0)) #quaternion 형태

            vr_transform, vr_rot = extract_vroot_transform(root_global_rot, root_global_pos) #T_virtual

            t_hip_global = translation_matrix(root_global_pos) @ glm.mat4_cast(root_global_rot) #T_hip
            t_local = glm.inverse(vr_transform) @ t_hip_global #T_virtual.inverse * T_hip

            root_local_pos = glm.vec3(t_local[3][0], t_local[3][1], t_local[3][2])
            root_local_rot = glm.quat_cast(t_local)

            #frame.hip_local_offsets = root_local_pos 
            frame.hip_local_offsets = glm.vec3(0, root_local_pos[1], 0) #잠시 루트 원점 고정
            frame.joint_rotations[root.name] = root_local_rot #hip의 rotation 값을 quaternion으로 저장
            frame.joint_local_transforms[root.name] = t_local
            #여기까지 hip에 대해서 재설정

            vr_pos = glm.vec3(vr_transform[3][0], vr_transform[3][1], vr_transform[3][2])

            frame.joint_positions["virtual_root"] = vr_pos #이건 global position이지 않겠는가?
            frame.joint_rotations["virtual_root"] = vr_rot
            frame.virtual_transform = vr_transform

            # self.compute_forward_kinematics(virtual_root, glm.mat4(1.0), frame)
            self.compute_forward_kinematics(root, vr_transform, frame)

    def compute_forward_kinematics(self, joint, parent_transform, motion_frame):
        """
        각 joint별로 Forward Kinemetic을 구현하기 위한 행렬입니다.
        Joint에 저장되어있는 local translation 과 rotation을 적용합니다.
        :param node: 적용할 Node (Joint)
        :param rotations: rotation 값
        :return: Forward Kinetic을 적용한 4x4 행렬
        """

        # 1. 현재 joint의 회전
        joint_rot = motion_frame.joint_rotations.get(joint.name, glm.quat(1, 0, 0, 0))
        R = glm.mat4_cast(joint_rot)

        # 2. joint의 local 위치 (default: offset)
        if joint.parent is None:
            T = glm.translate(glm.mat4(1.0), motion_frame.hip_local_offsets)  # virtual root은 global position을 갖고 있겠다.
                                                                                        # hip은 수정을 했으니 local 값을 가지고 있어야 한다.
        else:
            T = glm.translate(glm.mat4(1.0), glm.vec3(joint.offset))  # 그 외 joint는 이미 local offset을 가지고 있으니 괜찮다.

        # 3. local transform = T @ R
        local_transform = T @ R  # joint의 local transform을 가져와서 부모랑 곱해야지
        motion_frame.joint_local_transforms[joint.name] = local_transform

        # 4. global transform = parent @ local
        global_transform = parent_transform @ local_transform
        motion_frame.joint_global_transforms[joint.name] = global_transform

        # 5. global position 추출
        global_pos = glm.vec3(global_transform[3][0], global_transform[3][1], global_transform[3][2])
        #print("gp :", global_pos)
        motion_frame.joint_positions[joint.name] = global_pos
        #print(motion_frame.joint_positions[joint.name])
        # 6. 재귀적으로 하위 joint들도 처리
        for child in joint.children:
            self.compute_forward_kinematics(child, global_transform, motion_frame)

class MotionFrame:
    def __init__(self):
        self.joint_positions = {} #vec3으로 넣자. 당장 root만 position을 가짐
        self.joint_rotations = {} #Quaternion으로 들어가야함.
        self.hip_local_offsets = None
        self.joint_global_transforms = {}
        self.joint_local_transforms = {} #모든 관절의 local transform, T_local을 저장
        self.virtual_transform = None #virtual root transform


def bvh_parser(file_path):
    """
    BVH_Data를 받아서 parsing하는 함수입니다.
    이때 마지막 단계에서 virtual_root 노드를 추가로 더해 root Transform T로 사용합니다.
    :param file_path: 파일 경로
    """
    stack = []
    root = None
    cur_node = None
    motion_frames = []
    is_motion = False

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.split()
                if not parts:
                    continue

                if not is_motion:

                    if parts[0] == "MOTION":
                        is_motion = True

                    if parts[0] in ["ROOT", "JOINT", "End"]:
                        node = Joint(parts[1])
                        node.parent = cur_node
                        if not root:
                            root = node
                        if cur_node:
                            cur_node.children.append(node)

                        stack.append(node)
                        cur_node = node

                    elif parts[0] == "OFFSET":
                        cur_node.offset = list(map(float, parts[1:]))

                    elif parts[0] == "CHANNELS":
                        cur_node.channels = parts[2:]

                    elif parts[0] == "}":
                        stack.pop()
                        if stack:
                            cur_node = stack[-1]
                else:
                    if parts[0] == "Frames:":
                        frame_len = int(parts[1])
                        continue
                    elif parts[0] == "Frame":
                        print(parts[2])
                        frame_time = float(parts[2])
                        continue
                    frame_data = list(map(float, parts))
                    motion_frames.append(frame_data)

        motion_obj = Motion(motion_frames, frame_time, frame_len)
        motion_obj.list_to_quaternion(root)
        motion_obj.save_virtual_root_info(root)

        return root, motion_obj

    except FileNotFoundError:
        print(f"Error: File '{file_path}' does not exist.")

def check_bvh_structure(joint, is_root=False):
    """
    BVH_Data의 구조가 valid한지 재귀적으로 확인하는 함수입니다.
    :param joint: 확인할 joint
    """
    if joint.name == "virtual_root":
        if not joint.children:
            raise ValueError("Virtual root has no children.")
        for child in joint.children:
            check_bvh_structure(child, is_root=True)
        return
    if is_root:
        if len(joint.channels) != 6:
            raise ValueError(f"Root joint '{joint.name}' must have 6 channels, found {len(joint.channels)}")
        for channel in joint.channels[:3]:
            if "position" not in channel.lower():
                raise ValueError(
                    f"Root joint '{joint.name}' first three channels must be position channels, found '{channel}'")
        for channel in joint.channels[3:]:
            if "rotation" not in channel.lower():
                raise ValueError(
                    f"Root joint '{joint.name}' last three channels must be rotation channels, found '{channel}'")
    else:
        if joint.channels:
            if len(joint.channels) != 3:
                for channel in joint.channels[3:]:
                    if "rotation" not in channel.lower():
                        raise ValueError(f"Joint '{joint.name}' channel must be a rotation channel, found '{channel}'")
            else:
                for channel in joint.channels:
                    if "rotation" not in channel.lower():
                        raise ValueError(f"Joint '{joint.name}' channel must be a rotation channel, found '{channel}'")

    for child in joint.children:
        check_bvh_structure(child, is_root=False)

def process_joint(joint, frame, motion_frame, channel_idx):
    if joint.channels:
        num_channels = len(joint.channels)
        channels_val = frame[channel_idx : channel_idx + num_channels]

        quat = glm.quat(1,0,0,0)
        position = glm.vec3(0,0,0)

        if "position" in joint.channels[0].lower():
            for i, c in enumerate(joint.channels[:3]):
                val = channels_val[i]
                if "xposition" in c.lower():
                    position.x = val
                elif "yposition" in c.lower():
                    position.y = val
                elif "zposition" in c.lower():
                    position.z = val
            motion_frame.joint_positions[joint.name] = position
        # 회전 채널만 빼서 누적
            rot_channels = joint.channels[3:]
            rot_values = channels_val[3:]
        else:
            rot_channels = joint.channels
            rot_values = channels_val

            # 회전 채널 → Quaternion 누적
        for c, angle in zip(rot_channels, rot_values):
            theta = glm.radians(angle)
            if "xrotation" in c.lower():
                axis = glm.vec3(1, 0, 0)
            elif "yrotation" in c.lower():
                axis = glm.vec3(0, 1, 0)
            elif "zrotation" in c.lower():
                axis = glm.vec3(0, 0, 1)
            else:
                continue
            quat = quat * glm.angleAxis(theta, axis)

        quat = glm.normalize(quat)
        motion_frame.joint_rotations[joint.name] = quat
        channel_idx += num_channels

    for child in joint.children:
        channel_idx = process_joint(child, frame, motion_frame, channel_idx)

    return channel_idx


def motion_connect(motion1, motion2, root, transition_frames=60):
    m1_len = len(motion1.quaternion_frame)
    m2_len = len(motion2.quaternion_frame)

    m1_start_blend_idx = m1_len - transition_frames
    m2_start_blend_idx = 0

    m1_ref_frame = motion1.quaternion_frame[m1_start_blend_idx]
    m2_ref_frame = motion2.quaternion_frame[m2_start_blend_idx]

    pos1_vr = m1_ref_frame.joint_positions["virtual_root"]
    rot1_vr = m1_ref_frame.joint_rotations["virtual_root"]
    pos2_vr = m2_ref_frame.joint_positions["virtual_root"]
    rot2_vr = m2_ref_frame.joint_rotations["virtual_root"]

    rot_offset = rot1_vr * glm.conjugate(rot2_vr)
    pos_offset = pos1_vr - (rot_offset * pos2_vr)

    adjusted_m2_frames = []
    for frame in motion2.quaternion_frame:
        new_frame = MotionFrame()

        new_frame.joint_positions["virtual_root"] = (rot_offset * frame.joint_positions["virtual_root"]) + pos_offset
        new_frame.joint_rotations["virtual_root"] = rot_offset * frame.joint_rotations["virtual_root"]
        new_frame.hip_local_offsets = frame.hip_local_offsets
        new_frame.joint_global_transforms = frame.joint_global_transforms
        new_frame.joint_local_transforms = frame.joint_local_transforms
        new_frame.virtual_transform = frame.virtual_transform

        for name, rot in frame.joint_rotations.items():
            if name != "virtual_root":
                new_frame.joint_rotations[name] = rot
        adjusted_m2_frames.append(new_frame)

    final_motion_frames = motion1.quaternion_frame[:m1_start_blend_idx]

    blending_frames = []
    for i in range(transition_frames):
        alpha = (i + 1) / (transition_frames + 1)
        frame1 = motion1.quaternion_frame[m1_start_blend_idx + i]
        frame2 = adjusted_m2_frames[m2_start_blend_idx + i]
        blended_frame = MotionFrame()

        blended_frame.joint_positions["virtual_root"] = glm.mix(frame1.joint_positions["virtual_root"],frame2.joint_positions["virtual_root"], alpha)
        blended_frame.joint_rotations["virtual_root"] = glm.slerp(frame1.joint_rotations["virtual_root"],frame2.joint_rotations["virtual_root"], alpha)
        blended_frame.hip_local_offsets = glm.mix(frame1.hip_local_offsets, frame2.hip_local_offsets, alpha)
        for name, rot1 in frame1.joint_rotations.items():
            if name == "virtual_root": continue
            rot2 = frame2.joint_rotations.get(name)
            if rot2:
                if glm.dot(rot1, rot2) < 0:
                    rot2 = -rot2
                blended_frame.joint_rotations[name] = glm.slerp(rot1, rot2, alpha)
        blending_frames.append(blended_frame)

    final_motion_frames.extend(blending_frames)

    m2_remaining_start_idx = m2_start_blend_idx + transition_frames
    final_motion_frames.extend(adjusted_m2_frames[m2_remaining_start_idx:])

    new_motion = Motion(final_motion_frames, motion1.frame_time, len(final_motion_frames))

    frames_to_recalculate = new_motion.frames[m1_start_blend_idx:]

    for frame in frames_to_recalculate:
        vr_pos = frame.joint_positions.get("virtual_root", glm.vec3(0))
        vr_rot = frame.joint_rotations.get("virtual_root", glm.quat(1, 0, 0, 0))

        frame.virtual_transform = glm.translate(glm.mat4(1.0), vr_pos) @ glm.mat4_cast(vr_rot)
        new_motion.compute_forward_kinematics(root, frame.virtual_transform, frame)

    new_motion.quaternion_frame = new_motion.frames
    return new_motion