
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import math

def update_obs_space(env, delta):
    """
    更新多智能体环境中每个智能体的 Dict 观测空间。
    
    Args:
        env: 环境对象，observation_space 是 [Dict, Dict, ...]
        delta: 字典，键是字段名，值是新形状，例如 {'ship_vel': (3,), 'radar_obs': (6,)}
    
    Returns:
        更新后的 observation_space，类型为 [Dict, Dict, ...]
    """
    # 获取当前每个智能体的 Dict 模板（假设所有智能体相同）
    base_dict = env.observation_space[0].spaces.copy()
    
    # 更新或添加字段
    for key, shape in delta.items():
        base_dict[key] = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
    
    # 为每个智能体生成新的 Dict
    return [spaces.Dict(base_dict) for _ in range(len(env.observation_space))]

def distance_ss(own_ship, other_ship):
    """计算两点之间的距离
    """

    delta_x = other_ship.pos[0]-own_ship.pos[0]
    delta_y = other_ship.pos[1]-own_ship.pos[1]
    distance = (delta_x ** 2 + delta_y ** 2) ** 0.5

    return distance

def bearing_ss(own_ship, other_ship):
    """
    ownship随船坐标系的方位角度。,other_ship相对ownship的随船方位角度。
    """
    delta_x = other_ship.pos[0]-own_ship.pos[0]
    delta_y = other_ship.pos[1]-own_ship.pos[1]
    angle = np.atan2(delta_y, delta_x)
    bearing = angle if angle >= 0 else angle + 2 * np.pi
    bearing = (bearing - own_ship.pos[2])
    # bearing = bearing if bearing > 0 else bearing + 2 * np.pi
    bearing = normalize_angles(bearing)

    return bearing


def net(model):
    stats = {
        "weights": {},
        "biases": {}
    }

    for name, param in model.named_parameters():
        # 检查权重
        if "weight" in name:
            # 检测 NaN 和统计数据
            if torch.isnan(param.data).any():
                print(f"Warning: NaN detected in weights of layer {name}!")
                print(f"Values:\n{param.data}")
            #
            # weight_mean = param.data.mean().item()
            # weight_std = param.data.std().item()
            # stats["weights"][name] = {
            #     "mean": weight_mean,
            #     "std": weight_std,
            #     "values": param.data.cpu().numpy()  # 参数具体值
            # }
            # print(f"Layer: {name} | Weight Mean: {weight_mean} | Std: {weight_std}")
            # print(f"Values:\n{param.data}")

        # 检查偏置
        if "bias" in name:
            # 检测 NaN 和统计数据
            if torch.isnan(param.data).any():
                print(f"Warning: NaN detected in biases of layer {name}!")
                print(f"Values:\n{param.data}")
            #
            # bias_mean = param.data.mean().item()
            # bias_std = param.data.std().item()
            # stats["biases"][name] = {
            #     "mean": bias_mean,
            #     "std": bias_std,
            #     "values": param.data.cpu().numpy()  # 参数具体值
            # }
            # print(f"Layer: {name} | Bias Mean: {bias_mean} | Std: {bias_std}")
            # print(f"Values:\n{param.data}")

    return stats
def normalize_angles(angles):
    '''Puts angles in [-pi, pi] range.
    夹角也可以这么算，angel2-angle1，是相对angle1的夹角
    '''
    angles = angles.copy()
    if angles.size > 0:
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        assert -(np.pi + 1e-6) <= angles.min() and angles.max() <= (np.pi + 1e-6)
    return angles

def normalize_angles_torch(angles: torch.Tensor, validate: bool = False) -> torch.Tensor:
    """
    Puts angles into [-pi, pi].
    Equivalent to: (angles + pi) % (2*pi) - pi

    Args:
        angles: torch.Tensor of any shape/dtype float, on any device.
        validate: if True, assert the output range (will sync device).

    Returns:
        torch.Tensor on the same device/dtype as input.
    """
    # 不修改原张量
    out = torch.remainder(angles + math.pi, 2 * math.pi) - math.pi

    if validate and out.numel() > 0:
        # 为了断言需要取 .item()（会触发一次同步）
        amin = out.min().item()
        amax = out.max().item()
        tol = 1e-6
        assert -(math.pi + tol) <= amin and amax <= (math.pi + tol), \
            f"normalize_angles_torch: range check failed [{amin}, {amax}]"

    return out

def normalize_array(value_list):
    # 转换为 float 数组
    values = np.array(value_list, dtype=float).flatten()

    # 归一化
    min_value = np.min(values)
    max_value = np.max(values)

    value_list = (values - min_value) / (max_value - min_value)
    return value_list
