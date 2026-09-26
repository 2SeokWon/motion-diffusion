# core/config.py
import yaml
from types import SimpleNamespace


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    return obj


def load_config(path: str = "config.yml") -> SimpleNamespace:
    with open(path, 'r', encoding='utf-8') as f:  # Windows 기본 인코딩(cp949)으로 한글 주석을 읽다 깨지지 않도록
        raw = yaml.safe_load(f)

    # Compute derived model dimensions so scripts don't have to
    m = raw['model']
    m['joint_position_features'] = (m['njoints'] - 1) * m['position_features']  # 66
    m['joint_rotation_features'] = m['njoints'] * m['rotation_features']         # 138
    m['input_feats'] = (
        m['root_features'] +
        m['joint_position_features'] +
        m['joint_rotation_features'] +
        m['foot_features']
    )  # 210 (궤적은 생성 대상이 아니라 조건으로만 입력)
    m['cond_dim'] = m['cond_features'] + 1  # 4 = 경유점 값(x, z, yaw) + 마스크

    return _to_ns(raw)
