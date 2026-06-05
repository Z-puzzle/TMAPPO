from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import numpy as np
from envs.utils import *
import math
import torch

class CorpMac:
    def __init__(self,scheme, groups, args):

        self.args = args
        self.n_reds = args.n_reds
        self.n_blues = args.n_blues
        
        self.epsilon = self.args.epsilon #默认是0.2

        self.actions_space = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]


        #黑方参数
        self.black_detect_range = 30000.0  # 黑方对白方探测距离30km

    def init_hidden(self, batch_size):
        if self.args.agent in ("rnn","normalnn"):
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        else:
            pass

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs

        state = ep_batch["state"][:,t_ep]

        chosen_action = self.rule_policy(state,test_mode=test_mode,gain=self.args.gain)

        return chosen_action


    # 基于规则的蓝船策略，gain越大，说明调整的越快，理论上策略会更强。
    #没有apf的话，蓝船舵角是为0的。
    def rule_policy(self, state: torch.Tensor, test_mode: bool=False, gain: float=1.0):
        """
        state: [batch, state_dim] 或 [state_dim] 的 torch.Tensor（在 GPU 上）
        返回: [batch, n_blues] 的离散动作索引（torch.long, GPU）
        """
        device = state.device

        if state.dim() == 1:
            state = state.unsqueeze(0)

        n_ships = self.n_reds + self.n_blues
        feat_dim = state.shape[-1] // n_ships
        state = state.reshape(state.shape[0], n_ships, feat_dim)

        state_real = state.clone()
        state_real[..., 0:2] = state_real[..., 0:2] * 100.0
        state_real[..., 2] = state_real[..., 2] * (2.0 * math.pi)

        F_rep = self.apf_repulsive_force_tensor(state_real, self.n_reds, d0=100.0, eta=1.0)
        if getattr(self.args, "bound_apf_enable", True):
            F_bound = self.apf_boundary_force_tensor(state_real, d0=30.0, eta=1.0)
            F_total = F_rep + F_bound
        else:
            F_total = F_rep
        Fx, Fy = F_total[..., 0], F_total[..., 1]

        fi = state_real[:, self.n_reds:, 2]  # [batch, n_blues]

        eps = 1e-6
        F_norm = torch.linalg.norm(F_total, dim=-1)
        angle = torch.atan2(Fy, Fx)
        bearing = torch.remainder(angle, 2 * math.pi)
        angle_diff = normalize_angles_torch(bearing - fi)
        rudder = torch.clamp((angle_diff / math.pi) * gain, -1.0, 1.0)
        rudder = torch.where(F_norm < eps, torch.zeros_like(rudder), rudder)

        # 动作离散化到最近的动作槽
        actions = torch.as_tensor(self.actions_space, device=device, dtype=rudder.dtype)  # [A]
        diffs = torch.abs(rudder.unsqueeze(-1) - actions)  # [B, N, A]
        action_index = torch.argmin(diffs, dim=-1).to(torch.long)  # [B, N]

        return action_index

    def apf_boundary_force_tensor(self, obs, d0=300.0, eta=1.0):
        """
        obs   : (B, N_ships, obs_dim) 或 (N_ships, obs_dim)
        只对蓝船施加边界斥力，避免越界
        """
        device = obs.device
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        n_reds = self.n_reds
        blue_pos = obs[:, n_reds:, :2]  # (B, N_blue, 2)
        if blue_pos.shape[1] == 0:
            return torch.zeros(obs.shape[0], 0, 2, device=device)

        bounds = getattr(self.args, "bound_xy", [0.0, 500.0, -250.0, 250.0])
        x_min, x_max, y_min, y_max = map(float, bounds)

        x = blue_pos[..., 0]
        y = blue_pos[..., 1]
        eps = 1e-6

        Fx = torch.zeros_like(x)
        Fy = torch.zeros_like(y)

        # 左边界
        d = torch.clamp(x - x_min, min=eps)
        mask = d < d0
        coeff = eta * (1.0 / d - 1.0 / d0) * (1.0 / (d ** 2))
        Fx = Fx + torch.where(mask, coeff, torch.zeros_like(coeff))

        # 右边界
        d = torch.clamp(x_max - x, min=eps)
        mask = d < d0
        coeff = eta * (1.0 / d - 1.0 / d0) * (1.0 / (d ** 2))
        Fx = Fx - torch.where(mask, coeff, torch.zeros_like(coeff))

        # 下边界
        d = torch.clamp(y - y_min, min=eps)
        mask = d < d0
        coeff = eta * (1.0 / d - 1.0 / d0) * (1.0 / (d ** 2))
        Fy = Fy + torch.where(mask, coeff, torch.zeros_like(coeff))

        # 上边界
        d = torch.clamp(y_max - y, min=eps)
        mask = d < d0
        coeff = eta * (1.0 / d - 1.0 / d0) * (1.0 / (d ** 2))
        Fy = Fy - torch.where(mask, coeff, torch.zeros_like(coeff))

        F_bound = torch.stack([Fx, Fy], dim=-1)
        return F_bound
        
    def apf_repulsive_force_tensor(self, obs, n_reds, d0, eta=1.0):
        """
        obs   : (B, N_ships, obs_dim) 或 (N_ships, obs_dim) 的张量
        n_reds: 红船数量
        d0    : 斥力作用半径
        eta   : 斥力系数
        返回:
            F_rep: (B, N_blue, 2) 张量，每个蓝船受到的总斥力 [Fx, Fy]
        """
        device = obs.device
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        # 红船和蓝船位置
        red_pos = obs[:, :n_reds, :2]   # (B, N_red, 2)
        blue_pos = obs[:, n_reds:, :2]  # (B, N_blue, 2)

        if blue_pos.shape[1] == 0:
            return torch.zeros(obs.shape[0], 0, 2, device=device)

        # 向量：从红船指向蓝船，因为是斥力
        diff = blue_pos[:, :, None, :] - red_pos[:, None, :, :]  # (B, N_blue, N_red, 2)

        # 到蓝船的距离
        dist = torch.norm(diff, dim=-1)          # (B, N_blue, N_red)

        eps = 1e-6
        dist_safe = torch.clamp(dist, min=eps)

        # 只在 d < d0 内产生斥力
        mask = dist_safe < d0

        if not torch.any(mask):
            return torch.zeros(obs.shape[0], blue_pos.shape[1], 2, device=device)

        # 经典 APF 斥力公式:
        # F_i = η * (1/d - 1/d0) * (1/d^3) * (p_b - p_i)
        coeff = eta * (1.0 / dist_safe - 1.0 / d0) * (1.0 / (dist_safe ** 2))
        coeff = coeff * mask.to(coeff.dtype)

        # 合力
        F_rep = (diff * coeff.unsqueeze(-1)).sum(dim=2)  # (B, N_blue, 2)

        return F_rep
