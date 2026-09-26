# core/metrics.py
# 생성 결과 채점: 경유점 추종 + 발 미끄러짐. 단위는 BVH 데이터 그대로 cm (60fps).
import numpy as np

FOOT_JOINTS = ("RightToe", "LeftToe")

# 원본 데이터 측정 기준(2026-09-26): 딛고 있는 발끝 관절 높이는 1.5~2.5cm (수평 이동 중앙값 0.02cm/프레임),
# 3cm 이상은 공중에서 움직이는 발. 논문 기준 5cm는 이 데이터에서 원본도 37%가 미끄러진다고 판정되어 쓸 수 없다.
CONTACT_HEIGHT_CM = 2.5
# GMD/OmniControl의 20fps 기준 2.5cm를 60fps 프레임당 이동량으로 환산
SKATE_THRESHOLD_CM = 2.5 * 20 / 60


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def waypoint_metrics(gen_traj, gt_traj, mask):
    """
    생성 궤적이 경유점을 지났는지 채점한다. 단위: 위치 cm, 방향 deg.
    시작 프레임은 항상 원점이라 제외한다.
    full_path_ADE는 참고용: 경유점이 적으면 정답과 다른 길로 가는 것이 정상이다.
    """
    wp = np.flatnonzero(mask[:, 0])
    wp = wp[wp != 0]
    pos_err = np.linalg.norm(gen_traj[:, :2] - gt_traj[:, :2], axis=1)  # [T]
    yaw_err = np.abs(wrap_angle(gen_traj[:, 2] - gt_traj[:, 2]))       # [T]
    return {
        'num_waypoints': int(len(wp)),  # 도착 포함, 시작 제외
        'waypoint_pos_err_cm': float(pos_err[wp].mean()),
        'goal_pos_err_cm': float(pos_err[-1]),
        'waypoint_yaw_err_deg': float(np.degrees(yaw_err[wp].mean())),
        'full_path_ADE_cm': float(pos_err.mean()),
    }


def foot_positions(motion_obj, start=0, num_frames=None):
    """Motion 객체(FK 완료)에서 양 발끝의 전역 위치 [T, 2, 3]를 꺼낸다."""
    frames = motion_obj.quaternion_frame[start:None if num_frames is None else start + num_frames]
    return np.array([
        [[f.joint_positions[n].x, f.joint_positions[n].y, f.joint_positions[n].z] for n in FOOT_JOINTS]
        for f in frames
    ], dtype=np.float64)


def foot_skating(feet, contact_height=CONTACT_HEIGHT_CM, threshold=SKATE_THRESHOLD_CM):
    """
    발이 땅 근처(높이 < contact_height)인데 수평으로 움직인 정도.
    feet: [T, 2, 3] 발끝 전역 위치 (y가 높이)

    skating_ratio: 한쪽 발이라도 접촉 높이에서 threshold보다 많이 움직인 프레임 비율 (GMD/OmniControl 방식)
    weighted_skate_cm: 수평 이동 × (2 - 2^(h/H)) 의 프레임 평균. 바닥에 가까울수록 가중치 1,
                       접촉 높이에 가까울수록 0 → 발을 떼고 딛는 경계 순간의 영향이 줄어든다.
    """
    h = feet[1:, :, 1]                                                     # [T-1, 2]
    v = np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1)  # [T-1, 2] 수평 이동
    contact = h < contact_height
    weight = np.clip(2.0 - np.power(2.0, h / contact_height), 0.0, 1.0) * contact
    return {
        'skating_ratio': float((contact & (v > threshold)).any(axis=1).mean()),
        'weighted_skate_cm': float((v * weight).sum(axis=1).mean()),
        'contact_ratio': float(contact.mean()),
    }
