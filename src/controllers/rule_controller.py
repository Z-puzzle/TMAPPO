from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import numpy as np
from envs.utils import *
import math

class RuleMac:
    def __init__(self,scheme, groups, args):

        self.args = args

        self.n_agents = args.n_agents

        
        self.epsilon = self.args.epsilon #默认是0.2

        self.actions_space = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]


        #黑方参数
        self.black_detect_range = 30000.0  # 黑方对白方探测距离30km


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        if self.side == "red":
            obs = ep_batch["obs"][:,t_ep,:self.args.n_reds]
        else:
            obs = ep_batch["obs"][:,t_ep,self.args.n_reds:]
        chosen_action = self.rule_policy(obs,test_mode=test_mode,gain=self.args.gain)

        return chosen_action


    # 基于规则的蓝船策略，gain越大，说明调整的越快，理论上策略会更强。
    def rule_policy(self, obs: torch.Tensor, test_mode: bool=False, gain: float=1.0):
        """
        obs: [batch, n_agents, obs_dim] 的 torch.Tensor（在 GPU 上）
        返回: [batch, n_agents] 的离散动作索引（torch.long, GPU）
        """
        device = obs.device

        # 取特征（保持GPU）
        fi = obs[:, :, 2]   # 自身艏向
        dx = obs[:, :, 8]   # 对手相对 dx
        dy = obs[:, :, 9]   # 对手相对 dy

        # 角度计算（GPU）
        angle   = torch.atan2(dy, dx)                 # (-pi, pi]
        bearing = torch.remainder(angle, 2*math.pi)   # [0, 2pi)

        angle_diff = normalize_angles_torch(bearing - fi)
        rudder = torch.clamp((angle_diff / math.pi) * gain, -1.0, 1.0)  # [-1, 1]

        # 动作离散化到最近的动作槽
        actions = torch.as_tensor(self.actions_space, device=device, dtype=rudder.dtype)  # [A]
        diffs = torch.abs(rudder.unsqueeze(-1) - actions)  # [B, N, A]
        action_index = torch.argmin(diffs, dim=-1).to(torch.long)  # [B, N]

        # 可选：ε-greedy（仍在GPU）
        if not test_mode and getattr(self, "epsilon", 0) > 0:
            rand_idx = torch.randint(actions.numel(), size=action_index.shape, device=device)
            mask = (torch.rand_like(rudder) < self.epsilon)
            action_index = torch.where(mask, rand_idx, action_index)

        return action_index
        