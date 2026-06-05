import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .ship_gym import Ship


import json
import os
from .utils import *
import pandas as pd
import time
import torch as th


class ShipCombatEnv(gym.Env):
    """船舶作战环境"""

    def __init__(self,env_args):
        super().__init__()
        
        self.args=env_args
        self.n_reds = self.args['n_reds']
        self.n_blues = self.args['n_blues']
        self.n_ships = self.n_reds + self.n_blues
        self.n_reds_reward = self.args['n_reds_reward']
        self.n_blues_reward =self.args['n_blues_reward']


        self._build_ships()

        # 作战环境状态，红方win为1，蓝方win为-1，平均为0
        self.status = None
        self.start_pos = None
        self.replay_data = None  # 存储回放数据

        # 动作空间，舵角
        #连续
        self.action_space = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
        #离散
        # self.action_space = [spaces.Discrete(3) for _ in range(self.n_ships)]

        self.observation_space = [spaces.Dict({
            'ship_pos': spaces.Box(low=-500, high=500, shape=(3,), dtype=np.float64),
            'ship_vel': spaces.Box(low=-5, high=5, shape=(3,), dtype=np.float64),
            'health': spaces.Box(low=0, high=20, shape=(1,), dtype=np.float64),
        }) for _ in range(self.n_ships)]

        self.episode_limit = int(self.args.get("episode_limit", 256))

        self.n_actions = self.get_total_actions()

        self.start_pos = self.args["start_pos"]
        self.reward_terms = {}
        self.action_counts = {
            "red": np.zeros(self.n_actions, dtype=np.int64),
            "blue": np.zeros(self.n_actions, dtype=np.int64),
        }

    @staticmethod
    def _action_to_int(action):
        if action is None:
            return None
        if hasattr(action, "item"):
            try:
                return int(action.item())
            except Exception:
                pass
        try:
            return int(action)
        except Exception:
            return None

    @staticmethod
    def _typeid_onehot(typeid):
        onehot = np.zeros(3, dtype=np.float32)
        if typeid == 1:
            onehot[1] = 1.0
        elif typeid == -1:
            onehot[2] = 1.0
        return onehot


    def reset(self,seed=None, options=None):
        """重置环境，返回初始观测"""
        """先是包装器的reset,如果有包装器，会对这个obs进行处理包装""" 
        # 确保 _get_state() 返回与 observation_space 兼容的数据

        if options and self.start_pos is not None:
            for ship in self.ships:
                ship.reset()
            for i, ship in enumerate(self.ships):
                if i >= len(self.start_pos):
                    break
                ship.set_position(self.start_pos[i])
        else:
            for ship in self.ships:
                ship.reset()

            # 红船：优先在 x=0 这一列按 y=0, +10, -10, +20, -20 排列
            # 超过 5 条后，依次在 x=+10, -10, +20, -20 ... 的列继续同样的 y 排列
            red_spacing_x = float(self.args.get("red_start_spacing_x", 20.0))
            red_spacing_y = float(self.args.get("red_start_spacing_y", 50.0))
            red_center_x = float(self.args.get("red_start_center_x", 0.0))
            red_center_y = float(self.args.get("red_start_center_y", 0.0))
            red_heading = float(self.args.get("red_start_heading", 0.0))

            def _sym_offsets(spacing, count):
                vals = [0.0]
                k = 1
                while len(vals) < count:
                    vals.append(k * spacing)
                    if len(vals) < count:
                        vals.append(-k * spacing)
                    k += 1
                return vals

            x_cols = _sym_offsets(red_spacing_x, int(np.ceil(self.n_reds / 5)))
            y_order = _sym_offsets(red_spacing_y, 5)

            for i, ship in enumerate(self.red_ships):
                col = i // 5
                row = i % 5
                x = red_center_x + x_cols[col]
                y = red_center_y + y_order[row]
                ship.typeid = 1
                ship.health = self.args.get('max_health', 20.0)
                ship.set_position([x, y, red_heading])

            # 蓝船：在矩形区域内随机生成
            blue_x_min, blue_x_max = self.args.get("blue_start_x_range", [50.0, 250.0])
            blue_y_min, blue_y_max = self.args.get("blue_start_y_range", [-100.0, 100.0])
            for i, ship in enumerate(self.blue_ships):
                x = np.random.uniform(float(blue_x_min), float(blue_x_max))
                y = np.random.uniform(float(blue_y_min), float(blue_y_max))
                th = np.random.uniform(0.0, 2 * np.pi)
                ship.typeid = -1
                ship.health = self.args.get('max_health', 20.0)
                ship.set_position([x, y, th])

        
        obs = {}
        self.status = None
        self.t = 0
        self.reward_terms = {}
        self.action_counts["red"].fill(0)
        self.action_counts["blue"].fill(0)
        self.replay_data = {
            "obs":[],
            "actions":[],
            "rewards":[]
        }
        info = {}
        self.replay_data["obs"].append(self._get_state())
        return obs, info
    
    def _build_ships(self):

        self.red_ships=[]
        for i in range(self.n_reds):
            self.red_ships.append(Ship(f'red_{i}',args=self.args))
        for ship in self.red_ships:
            ship.typeid = 1

        self.blue_ships=[]
        for i in range(self.n_blues):
            self.blue_ships.append(Ship(f'blue_{i}',args=self.args))

        for ship in self.blue_ships:
            ship.typeid = -1

        self.ships=self.red_ships+self.blue_ships
        

    def _get_state(self):
        observations = []
        for i, ship in enumerate(self.ships):
            obs = {
                'ship_state': np.array([ship.pos[0], ship.pos[1], ship.pos[2], ship.vel[0],ship.vel[1],ship.vel[2],ship.health], dtype=np.float32),
            }
            observations.append(obs)
        return observations


    def step(self, actions):
        """执行一个动作，返回 (观测, 奖励, 是否结束, 额外信息);
            先执行这个函数，再接着处理包装器的observation的，
            逻辑是先自身环境的，再包装器中的，reset也是如此，先是环境中reset，再包装器中。
        """
        red_actions_raw = list(actions[0]) if len(actions) > 0 else []
        blue_actions_raw = list(actions[1]) if len(actions) > 1 else []

        replay_actions = [[], []]
        for i, action in enumerate(red_actions_raw):
            if i >= len(self.red_ships):
                break
            replay_actions[0].append(self._action_to_int(action))

        for i, action in enumerate(blue_actions_raw):
            if i >= len(self.blue_ships):
                break
            if getattr(self.blue_ships[i], "typeid", -1) == 0:
                replay_actions[1].append(None)
            else:
                replay_actions[1].append(self._action_to_int(action))

        self._record_actions(replay_actions)

        for i, action in enumerate(replay_actions[0]):
            if i >= len(self.red_ships):
                break
            if action is None:
                continue
            self.red_ships[i].move(tw=self.action_space[action])

        for i, action in enumerate(replay_actions[1]):
            if i >= len(self.blue_ships):
                break
            if action is None:
                continue
            self.blue_ships[i].move(tw=self.action_space[action], tx=2)

        self.replay_data["actions"].append(replay_actions)

        reward = np.zeros(self.n_reds_reward+self.n_blues_reward)  # 初始化奖励


        terminated = False
        # 返回当前的观测、奖励、是否结束以及额外信息
        self.t += 1
        truncated = self.t >= self.episode_limit and not terminated

        self.replay_data["obs"].append(self._get_state())
        return {}, reward, terminated, truncated, {}

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.n_actions,
                    "n_reds": self.n_reds,
                    "n_blues": self.n_blues,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_env(self):
        distances = []
        if self.n_reds > 0 and self.n_blues > 0:
            red_pos = np.array([ship.pos[:2] for ship in self.red_ships], dtype=np.float32)
            blue_pos = np.array([ship.pos[:2] for ship in self.blue_ships], dtype=np.float32)
            diff = red_pos[:, None, :] - blue_pos[None, :, :]
            distances = np.linalg.norm(diff, axis=-1).reshape(-1).tolist()
        min_red_blue_dist = float(min(distances)) if distances else float("nan")
        mean_red_blue_dist = float(np.mean(distances)) if distances else float("nan")

        return {
            "reward_terms": self.reward_terms,
            "action_counts_red": self.action_counts["red"].copy(),
            "action_counts_blue": self.action_counts["blue"].copy(),
            "min_red_blue_dist": min_red_blue_dist,
            "mean_red_blue_dist": mean_red_blue_dist,
        }
    

    def set_startpos(self,start_pos):
        self.start_pos = start_pos

    def close(self):
        """关闭环境"""
        pass
    
    def get_state_size(self):
        feats = self.args.get("state_features", ["pos", "vel", "typeid"])
        feature_sizes = {
            "pos": 3,
            "vel": 3,
            "health": 1,
            "typeid": 3,
        }
        per_ship = 0
        for feature in feats:
            if feature not in feature_sizes:
                raise KeyError(f"Unknown state feature: {feature}")
            per_ship += int(feature_sizes[feature])
        return self.n_ships * per_ship
    
    def get_obs_size(self):
        """动态计算单个智能体的观测维度"""
        obs_space = self.observation_space[0]  # 取第一个智能体的观测空间
        total_size = 0
        for key, space in obs_space.spaces.items():
            total_size += np.prod(space.shape)  # 计算每个子空间的元素总数
        return 9*(self.n_ships-1)+6
    
    def get_total_actions(self):
        """返回每个智能体的动作空间大小"""
        action_space = len(self.action_space)  # 取第一个智能体的动作空间
        return action_space

    def record_reward_term(self, name, red_value=0.0, blue_value=0.0):
        term = self.reward_terms.setdefault(name, {"red": 0.0, "blue": 0.0})
        term["red"] += float(red_value)
        term["blue"] += float(blue_value)

    def _record_actions(self, actions):
        red_actions_raw = actions[0] if len(actions) > 0 else []
        blue_actions_raw = actions[1] if len(actions) > 1 else []

        red_actions = []
        for action in red_actions_raw:
            action_int = self._action_to_int(action)
            if action_int is not None:
                red_actions.append(action_int)

        blue_actions = []
        for action in blue_actions_raw:
            action_int = self._action_to_int(action)
            if action_int is not None:
                blue_actions.append(action_int)

        red_actions = np.asarray(red_actions, dtype=np.int64).ravel()
        blue_actions = np.asarray(blue_actions, dtype=np.int64).ravel()
        red_actions = red_actions[(red_actions >= 0) & (red_actions < self.n_actions)]
        blue_actions = blue_actions[(blue_actions >= 0) & (blue_actions < self.n_actions)]
        if red_actions.size:
            self.action_counts["red"] += np.bincount(red_actions, minlength=self.n_actions)
        if blue_actions.size:
            self.action_counts["blue"] += np.bincount(blue_actions, minlength=self.n_actions)
    
    def as1d(self,x):
        a = np.asarray(x, dtype=np.float32)
    
        return a.reshape(-1)
        
    def get_state(self):
        """Returns the global state dynamically based on observation_space"""
        
        self.FEAT = {
            "pos":    lambda ship: self.as1d([ship.pos[0] / 100.0, ship.pos[1] / 100.0, ship.pos[2] / (2.0 * np.pi)]),
            "vel":    lambda ship: self.as1d(ship.vel),
            "health": lambda ship: self.as1d(getattr(ship, "health") / 20.0),
            "typeid": lambda ship: self._typeid_onehot(getattr(ship, "typeid")),
        }        

        feats = self.args.get("state_features", ["pos", "vel", "typeid"])
        rows = []
        for ship in self.ships:
            row = np.concatenate([self.FEAT[f](ship) for f in feats], axis=0).astype(np.float32)
            if getattr(ship, "typeid", -1) == 0:
                row[:] = 0.0
            rows.append(row)

        state = np.concatenate(rows, axis=0).astype(np.float32)

        return state
    
    # 示例：根据环境状态动态设置某些动作不可用（可根据实际逻辑修改）
    def get_avail_actions(self):
        """Returns available actions for all agents in a continuous action space"""
        avail_actions = np.ones((self.n_ships, self.n_actions), dtype=np.int64)
            
            
            # 假设第二船的动作 1（0.0）在某些情况下不可用
            # if some_condition:  # 替换为实际环境状态检查
            #     avail_actions[1, 1] = 0  # 第二船的动作 1 不可用
    
        return avail_actions
        
    def get_flatten_obs(self, obs):
        """flattened observations"""

        values = [value for key, value in obs.items()]

        concatenated_obs = np.concatenate(values, axis=1)

        flatten_obs = np.zeros((self.n_ships,concatenated_obs.shape[-1]))
        flatten_obs[:concatenated_obs.shape[0]] = concatenated_obs

        return flatten_obs
    
    def save_replay(self):
        if self.replay_data is None:
            print("No replay data to save. Run environment first.")
            return
        
        save_dir = os.path.join(self.args['replay_dir'])
        os.makedirs(save_dir, exist_ok=True)

        # 使用时间戳或其他唯一标识生成文件名
        unique_id = time.strftime("%Y%m%d_%H%M%S")  # 示例：20250422_143022
        file_name = f"data_{unique_id}.csv"  # 唯一文件名
        file_path = os.path.join(save_dir, file_name)

        # Define CSV columns dynamically
        columns = ["step"]
        for i in range(self.n_reds):
            columns.append(f"pos_red{i+1}")
            columns.append(f"health_red{i+1}")
            columns.append(f"action_red{i+1}")
        for i in range(self.n_blues):
            columns.append(f"pos_blue{i+1}")
            columns.append(f"health_blue{i+1}")
            columns.append(f"action_blue{i+1}")

        # Initialize CSV file with header if it doesn't exist
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if not os.path.exists(file_path):
                df = pd.DataFrame(columns=columns)
                df.to_csv(file_path, index=False)
        except Exception as e:
            print(f"Failed to initialize CSV file: {e}")
            return

        # Prepare data for all time steps
        data_list = []
        n_steps = len(self.replay_data["obs"])
        for t in range(n_steps):
            data = {"step": t}

            # Handle observations
            obs_t = self.replay_data["obs"][t]
            for i in range(self.n_reds):
                obs = obs_t[i]
                # Save as string to handle lists (e.g., "[0.1,0.2]")
                data[f"pos_red{i+1}"] = obs['ship_state'][0:3]
                data[f"health_red{i+1}"] = obs['ship_state'][6]
            for i in range(self.n_blues):
                obs = obs_t[i+self.n_reds]
                # Save as string to handle lists (e.g., "[0.1,0.2]")
                data[f"pos_blue{i+1}"] = obs['ship_state'][0:3]
                data[f"health_blue{i+1}"] = obs['ship_state'][6]

            # Handle actions (fill None if empty or insufficient)
            if t < len(self.replay_data["actions"]) and self.replay_data["actions"][t] is not None:
                actions_t = self.replay_data["actions"][t]
                for i in range(self.n_reds):
                    action = actions_t[0][i] if i < len(actions_t[0]) else None
                    data[f"action_red{i+1}"] = self._action_to_int(action)
                for i in range(self.n_blues):
                    action = actions_t[1][i] if i < len(actions_t[1]) else None
                    data[f"action_blue{i+1}"] = self._action_to_int(action)
            else:
                for i in range(self.n_ships):
                    data[f"action_agent{i+1}"] = None

            data_list.append(data)

        # Write all data to CSV at once
        try:
            df = pd.DataFrame(data_list, columns=columns)
            df.to_csv(file_path, mode='a', index=False, header=False, float_format="%.4f")
            print(f"Replay saved at: {file_path}")
        except Exception as e:
            print(f"Failed to write to CSV: {e}")


# 测试自定义环境
#
def make_env(args):
    
    from .observation_wrapper import FireObs,RadarObs,OwnObs
    from .reward_wrapper import TerminatedWrapper,FieldRewardWrapper,HealthRewardWrapper,PBRSRewardWrapper

    env = ShipCombatEnv(env_args=args)

    # env.start_pos = env.generate_startpos()
    # env.reset()

    # 观测，obswrapper从上到下
    env = OwnObs(env)
    env = FireObs(env) #观测到，扣除生命值
    env = RadarObs(env)

    # 奖励，先主环境step，再rew添加从上到下
    # env = OutRewardWrapper(env)
    # env = HealthRewardWrapper(env)
    # 没有航行惩罚奖励，船容易一直躲着敌船，timeout，但有了这个，奖励的总和就不确定。
    # env = NavigateWrapper(env)
    if args["field_reward"]:
        env = FieldRewardWrapper(env)
    else:
        env = HealthRewardWrapper(env)
    if args["PBRS"]:
        env = PBRSRewardWrapper(env)
    # env = CollisionWrapper(env)

    # 终止奖励，一定要放到最后
    env = TerminatedWrapper(env)

    env.reset()

    return env

def make_hunt_env(args):
    from .observation_wrapper_hunting import RadarObs,OwnObs
    from .reward_wrapper_hunting import TerminatedWrapper,GuideRewardWrapper,CollisionWrapper

    env = ShipCombatEnv(env_args=args)

    
    # 观测，obswrapper从上到下
    # env = OwnObs(env)
    env = RadarObs(env)

    # 奖励，先主环境step，再rew添加从上到下
    # env = OutRewardWrapper(env)
    # env = HealthRewardWrapper(env)
    # 没有航行惩罚奖励，船容易一直躲着敌船，timeout，但有了这个，奖励的总和就不确定。
    # env = NavigateWrapper(env)

    if args.get("collision_penalty", False):
        env = CollisionWrapper(env)
    env= GuideRewardWrapper(env)
    # env = TimeStepWrapper(env)

    # 终止奖励，一定要放到最后
    env = TerminatedWrapper(env)

    env.reset()

    return env

if __name__ == '__main__':
    from main import config
    args = config()
    env = make_env(args)
    # It will check your custom environment and output additional warnings if needed
