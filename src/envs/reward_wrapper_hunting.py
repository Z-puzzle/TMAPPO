import gymnasium as gym
import numpy as np
from .utils import *


class TerminatedWrapper(gym.Wrapper):
    def __init__(self, env, reward_scale=1.0, encircle_dist=30, min_encircle_ships=2, encirclement_k=10.0,encircle_std_thresh=None):
        super().__init__(env)
        self.n_reds = self.unwrapped.n_reds
        self.n_blues = self.unwrapped.n_blues
        self.n_ships = self.unwrapped.n_ships
        self.ships = self.unwrapped.ships
        self.reward_scale = reward_scale
        self.red_ships = self.unwrapped.red_ships
        self.blue_ships = self.unwrapped.blue_ships

        self.n_reds_reward = self.unwrapped.n_reds_reward

        self.episode_limit = self.unwrapped.episode_limit
        self.min_encircle_ships = min_encircle_ships
        self.encircle_dist = encircle_dist
        self.encirclement_k = encirclement_k
        args = getattr(self.unwrapped, "args", {}) or {}
        self.encircle_dense_coef = float(args.get("encircle_dense_coef", 30.0))
        self.encircle_dense_clip = args.get("encircle_dense_clip", 0.1)
        self.encircle_dense_eps = float(args.get("encircle_dense_eps", 1e-6))
        encircle_std_thresh = args.get("encircle_std_thresh", 1)
        self.encircle_std_thresh = None if encircle_std_thresh is None else float(encircle_std_thresh)
        self.prev_encircle_score_by_blue = {}
        self.prev_inside_count_by_blue = {}

    #如果蓝船被围住，就终止。这里可能后面要修改，包围蓝船不一定是终止，蓝船只是死亡，还有别的蓝船。
    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)

        r_capture = self.encirclement_k * 2 * self.reward_scale
        for blue_idx, blue_ship in enumerate(self.blue_ships):
            if getattr(blue_ship, "typeid", -1) == 0:
                continue

            enc = self.compute_encirclement(blue_ship, self.encircle_dist)
            inside_idx = enc["inside_idx"]

            if enc["encircled"]:
                blue_ship.typeid = 0
                if self.encircle_dense_coef == 0.0:
                    continue

            inside_count = int(len(inside_idx))
            sigma_ratio = enc.get("sigma_ratio")
            score = float(math.exp(-sigma_ratio)) if sigma_ratio is not None and inside_count >= 2 else 0.0

            prev_inside_count = self.prev_inside_count_by_blue.get(blue_idx)
            prev_score = float(self.prev_encircle_score_by_blue.get(blue_idx, 0.0))
            self.prev_inside_count_by_blue[blue_idx] = inside_count
            self.prev_encircle_score_by_blue[blue_idx] = float(score)

            if prev_inside_count is None or inside_count != prev_inside_count:
                continue

            delta = max(0.0, float(score) - prev_score)
            dense_reward = self.encircle_dense_coef * delta * self.reward_scale
            if self.encircle_dense_clip is not None:
                dense_reward = float(np.clip(dense_reward, -self.encircle_dense_clip, self.encircle_dense_clip))

            if self.n_reds_reward == 1:
                rew[:self.n_reds_reward] += dense_reward
            elif len(inside_idx) > 0:
                rew[inside_idx] += dense_reward

        if all(getattr(ship, "typeid", -1) == 0 for ship in self.blue_ships):
            terminated = True

        return obs, rew, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_encircle_score_by_blue = {}
        self.prev_inside_count_by_blue = {}
        return obs, info
    
    def get_state(self):
        return self.unwrapped.get_state()

    def get_avail_actions(self):
        return self.unwrapped.get_avail_actions()
    
    def get_env(self):
        return self.unwrapped.get_env()

    def get_flatten_obs(self, obs):
        return self.unwrapped.get_flatten_obs(obs)
    
    def get_env_info(self):
        return self.unwrapped.get_env_info()

    def save_replay(self):
        return self.unwrapped.save_replay()
    
    def compute_encirclement(self, blue_ship, d_cap):
        """
        self.redships   : list of ship objects
        blue_ship       : target ship M
        d_cap           : 围捕半径
        """
        if blue_ship is None:
            return {
                "inside_idx": np.array([], dtype=int),
                "sorted_idx": None,
                "angles": None,
                "gaps": None,
                "max_gap": None,
                "sigma_ratio": None,
                "encircled": False,
            }

        red_pos = np.array([ship.pos[:2] for ship in self.red_ships], dtype=np.float32)
        blue_pos = np.array(blue_ship.pos[:2], dtype=np.float32)
        distances = np.linalg.norm(red_pos - blue_pos[None, :], axis=1)
        inside_idx = np.where(distances <= d_cap)[0]

        if len(inside_idx) < 2:
            return {
                "inside_idx": inside_idx,
                "sorted_idx": None,
                "angles": None,
                "gaps": None,
                "max_gap": None,
                "sigma_ratio": None,
                "encircled": False,
            }

        mx, my = float(blue_ship.pos[0]), float(blue_ship.pos[1])
        angles = np.array(
            [math.atan2(float(self.red_ships[idx].pos[1]) - my, float(self.red_ships[idx].pos[0]) - mx) for idx in inside_idx],
            dtype=np.float32,
        )
        order = np.argsort(angles)
        sorted_idx = inside_idx[order]
        sorted_angles = angles[order]

        gaps = np.diff(sorted_angles, append=float(sorted_angles[0]) + 2 * math.pi)
        max_gap = float(gaps.max())

        ideal_gap = (2 * math.pi) / float(len(inside_idx))
        ratio = np.asarray(gaps, dtype=np.float32) / float(ideal_gap + self.encircle_dense_eps)
        sigma_ratio = float(np.std(ratio))

        encircled = bool(
            len(inside_idx) >= self.min_encircle_ships
            and max_gap <= math.pi
            and (self.encircle_std_thresh is None or sigma_ratio <= self.encircle_std_thresh)
        )

        return {
            "inside_idx": inside_idx,
            "sorted_idx": sorted_idx,
            "angles": sorted_angles,
            "gaps": gaps,
            "max_gap": max_gap,
            "sigma_ratio": sigma_ratio,
            "encircled": encircled,
        }
    
class GuideRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_scale=1.0, d_cap=25, progress_coef=0.5,guide_heading_coef=0.0,turn_penalty_coef=0.0, guide_progress_reward_clip=0.5,guide_heading_reward_clip=0.1):
        super().__init__(env)
        self.n_agents = getattr(self.env.unwrapped, "n_agents", 1)
        self.n_reds = self.unwrapped.n_reds
        self.n_blues = self.unwrapped.n_blues
        self.n_reds_reward = self.unwrapped.n_reds_reward
        self.red_ships = self.unwrapped.red_ships
        self.blue_ships = self.unwrapped.blue_ships

        self.reward_scale = reward_scale
        args = getattr(self.unwrapped, "args", {}) or {}
        self.d_cap = float(args.get("guide_d_cap", d_cap))
        self.progress_coef = float(args.get("guide_progress_coef", progress_coef))
        self.heading_coef = float(args.get("guide_heading_coef", guide_heading_coef))
        self.turn_penalty_coef = float(args.get("guide_turn_penalty_coef", turn_penalty_coef))
        self.progress_reward_clip = args.get("guide_progress_reward_clip", guide_progress_reward_clip)
        self.heading_reward_clip = args.get("guide_heading_reward_clip", guide_heading_reward_clip)

        self.action_space = getattr(self.unwrapped, "action_space", None)

        self.prev_distances = None
        self.prev_has_target = False

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_distances, _, self.prev_has_target = self._compute_red_distances()
        return obs, info

    def step(self, actions):
        obs, rew, terminated, truncated, info = self.env.step(actions)

        current_distances, nearest_bearing, has_target = self._compute_red_distances()
        if self.prev_distances is None:
            self.prev_distances = current_distances
            self.prev_has_target = has_target

        # Potential-based shaping: phi = -|dist - d_cap|
        # Reward is delta phi so "staying on the ring" won't accumulate.
        if self.prev_has_target and has_target:
            phi_prev = -np.abs(self.prev_distances - self.d_cap)
            phi_curr = -np.abs(current_distances - self.d_cap)
            progress_reward = (phi_curr - phi_prev) * self.progress_coef * self.reward_scale
            if self.progress_reward_clip is not None:
                progress_reward = np.clip(progress_reward, -self.progress_reward_clip, self.progress_reward_clip)
        else:
            progress_reward = np.zeros(self.n_reds, dtype=np.float32)

        self.prev_distances = current_distances
        self.prev_has_target = has_target

        heading_reward = np.zeros(self.n_reds, dtype=np.float32)
        if self.heading_coef != 0.0:
            heading = np.array([ship.pos[2] for ship in self.red_ships], dtype=np.float32)
            err = np.arctan2(np.sin(nearest_bearing - heading), np.cos(nearest_bearing - heading))
            heading_reward = np.cos(err).astype(np.float32) * self.heading_coef * self.reward_scale
            if self.heading_reward_clip is not None:
                heading_reward = np.clip(heading_reward, -self.heading_reward_clip, self.heading_reward_clip)

        turn_penalty = np.zeros(self.n_reds, dtype=np.float32)
        if self.turn_penalty_coef != 0.0 and self.action_space is not None:
            try:
                red_actions = np.asarray(actions[0], dtype=np.int64).reshape(-1)
            except Exception:
                red_actions = np.zeros(self.n_reds, dtype=np.int64)
            red_actions = red_actions[: self.n_reds]
            for i, a in enumerate(red_actions):
                if 0 <= int(a) < len(self.action_space):
                    tw = float(self.action_space[int(a)])
                else:
                    tw = 0.0
                turn_penalty[i] = float(-abs(tw) * self.turn_penalty_coef * self.reward_scale)

        if self.n_reds_reward == 1:
            rew[:self.n_reds_reward] += progress_reward.mean() + heading_reward.mean() + turn_penalty.mean()
        else:
            count = min(self.n_reds_reward, self.n_reds)
            rew[:count] += progress_reward[:count] + heading_reward[:count] + turn_penalty[:count]

        return obs, rew, terminated, truncated, info
    
    #这里是计算最近的蓝船距离
    def _compute_red_distances(self):
        red_pos = np.array([ship.pos[:2] for ship in self.red_ships], dtype=np.float32)
        active_blues = [ship for ship in self.blue_ships if getattr(ship, "typeid", -1) != 0]
        blue_pos = np.array([ship.pos[:2] for ship in active_blues], dtype=np.float32)
        if blue_pos.size == 0 or red_pos.size == 0:
            distances = np.zeros(self.n_reds, dtype=np.float32)
            bearing = np.zeros(self.n_reds, dtype=np.float32)
            return distances, bearing, False
        diff = blue_pos[None, :, :] - red_pos[:, None, :]
        dists = np.linalg.norm(diff, axis=-1)
        distances = dists.min(axis=1).astype(np.float32)
        nearest = np.argmin(dists, axis=1)
        idx = np.arange(diff.shape[0])
        dx = diff[idx, nearest, 0]
        dy = diff[idx, nearest, 1]
        bearing = np.arctan2(dy, dx).astype(np.float32)
        return distances, bearing, True

 

class CollisionWrapper(gym.Wrapper):
    def __init__(self, env, reward_scale=1.0, r_max=6.0, penalty_k=3.0,collision_penalty_clip=6.0):
        super().__init__(env)
        self.n_reds = self.unwrapped.n_reds
        self.n_blues = self.unwrapped.n_blues
        self.n_reds_reward = self.unwrapped.n_reds_reward
        self.n_blues_reward = self.unwrapped.n_blues_reward
        self.n_ships = self.unwrapped.n_ships
        self.ships = self.unwrapped.ships
        self.reward_scale = reward_scale
        args = getattr(self.unwrapped, "args", {}) or {}

        # 碰撞惩罚参数
        self.r_max = float(args.get("collision_r_max", r_max))  # 只在该半径内惩罚
        # 距离过近的惩罚强度（k >= 0）
        # 兼容旧参数 collision_penalty_value（可能为负），内部取 abs 作为 k
        k = args.get("collision_penalty_k", penalty_k)
        self.penalty_k = float(abs(k))
        penalty_clip = args.get("collision_penalty_clip", collision_penalty_clip)
        self.penalty_clip = None if penalty_clip is None else float(abs(penalty_clip))

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        all_pos = np.array([ship.pos[:2] for ship in self.ships], dtype=np.float32)
        delta_pos = all_pos[:, None, :] - all_pos[None, :, :]
        dist_matrix = np.linalg.norm(delta_pos, axis=-1).astype(np.float32)
        np.fill_diagonal(dist_matrix, np.inf)
        d_agent = dist_matrix.min(axis=1).astype(np.float32)
        d_min = d_agent[: self.n_reds]
        # if np.any(d_min <= self.r_max):
        #     breakpoint()

        # 碰撞惩罚（连续变强）：
        # penalty(d) = 0                        if d >= r_max
        # penalty(d) = -k * (r_max - d)^2        if d < r_max
        delta = np.maximum(self.r_max - d_min, 0.0).astype(np.float32)
        red_penalties = (-self.penalty_k * (delta ** 2)).astype(np.float32)
        if self.penalty_clip is not None:
            red_penalties = np.clip(red_penalties, -self.penalty_clip, self.penalty_clip).astype(np.float32)

        # 将惩罚加回 reward（全局/个体模式由 n_reds_reward 控制）
        avg_red_penalty = np.sum(red_penalties) / max(1, self.n_reds)
        if self.n_reds_reward == 1:
            rew[:self.n_reds_reward] += avg_red_penalty * self.reward_scale
        else:
            count = min(self.n_reds_reward, self.n_reds)
            rew[:count] += red_penalties[:count] * self.reward_scale
        # 下面是给所有船加惩罚：
        # rew += collision_penalty * self.reward_scale

        return obs, rew, terminated, truncated, info
