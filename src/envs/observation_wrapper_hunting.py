import gymnasium as gym
import numpy as np
from .utils import *

# 是否在火力打击范围内,这里没有加障碍物遮挡，后面可以仿照openai，加上障碍物遮挡


# 观测范围内船舶的观测信息,这里只考虑了单船
class RadarObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_ships = self.unwrapped.n_ships
        self.n_reds = self.unwrapped.n_reds

        self.red_ships = self.unwrapped.red_ships
        self.blue_ships = self.unwrapped.blue_ships
        self.ships = self.unwrapped.ships


        # 每个目标船 11 维特征；最终每个红方智能体观测为 (n_ships * 11,)
        self.ship_features = [
            "delta_x",
            "delta_y",
            "delta_phi",
            "distance",
            "bearing_rel",
            "vx",
            "vy",
            "r",
            "is_self",
            "is_friend",
            "is_enemy",
        ]
        self.unwrapped.observation_space = update_obs_space(
            env, {"radar_obs": (self.n_ships * len(self.ship_features),)}
        )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs),info

    #全特征拼接obs
    def observation(self, obs):
        ships = self.ships
        reds = self.red_ships
        n_reds = self.n_reds
        n_ships = self.n_ships

        all_p = np.array([s.pos for s in ships])
        all_v = np.array([s.vel for s in ships])
        all_t = np.array([s.typeid for s in ships])
        red_p = np.array([s.pos for s in reds])
        red_v = np.array([s.vel for s in reds])

        ship_to_idx = {id(s): idx for idx, s in enumerate(ships)}
        self_idx = np.array([ship_to_idx[id(s)] for s in reds], dtype=np.int64)

        rel_pos = all_p[np.newaxis, :, :2] - red_p[:, np.newaxis, :2]
        dx = rel_pos[:, :, 0]
        dy = rel_pos[:, :, 1]
        dist = np.linalg.norm(rel_pos, axis=2)
        bearing = np.arctan2(dy, dx)

        phi_self = red_p[:, 2:3]  # (M, 1)
        phi_target = np.broadcast_to(all_p[:, 2], (n_reds, n_ships))
        delta_phi = normalize_angles(phi_target - phi_self)
        bearing_rel = normalize_angles(bearing - phi_self)

        rel_v = all_v[np.newaxis, :, :] - red_v[:, np.newaxis, :]
        vx, vy, r = rel_v[:, :, 0], rel_v[:, :, 1], rel_v[:, :, 2]


        is_friend = all_t == 1
        is_enemy = all_t == -1
        is_inactive = all_t == 0
        t_onehot = np.zeros((n_reds, n_ships, 3), dtype=np.float32)
        t_onehot[:, is_friend, 1] = 1.0
        t_onehot[:, is_enemy, 2] = 1.0
        t_onehot[np.arange(n_reds), self_idx] = (1.0, 0.0, 0.0)

        base_feats = np.stack(
            [
                dx / 100.0,
                dy / 100.0,
                delta_phi / np.pi,
                dist / 100.0,
                bearing_rel / np.pi,
                vx,
                vy,
                r,
            ],
            axis=-1,
        )
        base_feats[np.arange(n_reds), self_idx, :] = 0.0
        if np.any(is_inactive):
            base_feats[:, is_inactive, :] = 0.0
            t_onehot[:, is_inactive, :] = 0.0

        feats = np.concatenate([base_feats, t_onehot], axis=-1)
        full_obs = feats.reshape(n_reds, -1).astype(np.float32)

        obs['radar_obs'] = full_obs
        return obs

class GridRadarObs(gym.ObservationWrapper):
    def __init__(self, env, grid_size=(10, 10), grid_range=None, ego_frame=True, key="radar_grid"):
        super().__init__(env)
        self.n_ships = self.unwrapped.n_ships
        self.n_reds = self.unwrapped.n_reds

        self.red_ships = self.unwrapped.red_ships
        self.blue_ships = self.unwrapped.blue_ships
        self.ships = self.unwrapped.ships

        self.grid_rows, self.grid_cols = grid_size
        if grid_range is None:
            args = getattr(self.unwrapped, "args", {})
            grid_range = args.get("radar_range", 100.0)
        self.grid_range = float(grid_range)
        self.ego_frame = ego_frame
        self.key = key

        self.cell_w = (2.0 * self.grid_range) / self.grid_cols
        self.cell_h = (2.0 * self.grid_range) / self.grid_rows
        self.channels = 2  # [red_count, blue_count]

        self.unwrapped.observation_space = update_obs_space(
            env, {self.key: (self.grid_rows * self.grid_cols * self.channels,)}
        )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def _to_ego(self, dx, dy, heading):
        if not self.ego_frame:
            return dx, dy
        c = np.cos(-heading)
        s = np.sin(-heading)
        return dx * c - dy * s, dx * s + dy * c

    def observation(self, obs):
        grid_obs = np.zeros(
            (self.n_reds, self.grid_rows, self.grid_cols, self.channels), dtype=np.float32
        )

        for i, red_ship in enumerate(self.red_ships):
            for ship in self.ships:
                if ship is red_ship:
                    continue
                dx = ship.pos[0] - red_ship.pos[0]
                dy = ship.pos[1] - red_ship.pos[1]
                dx, dy = self._to_ego(dx, dy, red_ship.pos[2])

                if abs(dx) > self.grid_range or abs(dy) > self.grid_range:
                    continue

                col = int((dx + self.grid_range) / self.cell_w)
                row = int((dy + self.grid_range) / self.cell_h)
                if col < 0 or row < 0 or col >= self.grid_cols or row >= self.grid_rows:
                    continue

                ch = 0 if ship.typeid == 0 else 1
                grid_obs[i, row, col, ch] += 1.0

        obs[self.key] = grid_obs.reshape(self.n_reds, -1)
        return obs

class OwnObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_ships = self.unwrapped.n_ships
        self.n_reds = self.unwrapped.n_reds

        self.red_ships = self.unwrapped.red_ships
        self.blue_ships = self.unwrapped.blue_ships

        # self.unwrapped.observation_space = update_obs_space(env, {'radar_obs': (len(self.features),)})

    def reset(self,seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs),info


    def observation(self, obs):
        # obs['radar_obs'] = np.zeros((self.n_ships, 8))

        # 红对蓝的探测
        own_obs = []
        for red_ship in self.red_ships:

            own_obs.append([red_ship.pos/100,red_ship.vel])

        own_obs = np.array(own_obs).reshape(self.n_reds,-1)

        obs['ship_state'] = own_obs

        return obs
