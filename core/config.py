# core/config.py
import yaml
from types import SimpleNamespace


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    return obj


def load_config(path: str = "config.yml") -> SimpleNamespace:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    # Compute derived model dimensions so scripts don't have to
    m = raw['model']
    m['joint_position_features'] = (m['njoints'] - 1) * m['position_features']  # 66
    m['joint_rotation_features'] = m['njoints'] * m['rotation_features']         # 138
    m['input_feats'] = (
        m['root_features'] +
        m['joint_position_features'] +
        m['joint_rotation_features'] +
        m['foot_features'] +
        m['cond_features']
    )  # 213

    return _to_ns(raw)
