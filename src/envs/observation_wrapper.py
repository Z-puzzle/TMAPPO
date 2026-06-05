import gymnasium as gym
import numpy as np
from .utils import *

# 是否在火力打击范围内,这里没有加障碍物遮挡，后面可以仿照openai，加上障碍物遮挡
class FireObs(gym.Wrapper):
    """ Adds an mask observation that states which agents are visible to which agents.
        Args:
            cone_angle: (float) the angle in radians btw the axis and edge of the observation cone
    """
    def __init__(self, env):
        super().__init__(env)
        self.n_ships = self.unwrapped.n_ships
        self.ships = self.unwrapped.ships
        self.cone_angle = self.ships[0].fire_angle


    def reset(self,seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs,info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)

        # 智能体的打击mask
        agent_pos2d = obs['ship_state'][:,:2]  # (n_ships, 2)
        agent_angle = obs['ship_state'][:,2] 
        cone_mask = self.in_cone2d(agent_pos2d, agent_angle, self.cone_angle, agent_pos2d)

        # 初始化每艘船的伤害
        damages = np.zeros(self.n_ships)

        # i对j的探测
        for i in range(self.n_ships):
            for j in range(self.n_ships):
                if i != j:
                    distance = distance_ss(self.ships[i], self.ships[j])
                    # 方位在火力打击范围内
                    if distance < self.ships[i].fire_range and cone_mask[i, j]:
                        cone_mask[i, j] = True
                        # 记录j船受到的伤害
                        damages[j] += 1.0
                    else:
                        cone_mask[i, j] = False

        # 应用伤害并更新生命值
        for j in range(self.n_ships):
            new_health = self.ships[j].health - damages[j]
            self.ships[j].update_health(new_health)

        info['fire_obs'] = cone_mask

        return obs, rew, terminated, truncated, info

    def in_cone2d(self, origin_pts, origin_angles, cone_angle, target_pts):
        '''
            Computes whether 2D points target_pts are in the cones originating from
                origin_pts at angle origin_angles with cone spread angle cone_angle.
            Args:
                origin_pts (np.ndarray): array with shape (n_points, 2) of origin points
                origin_angles (np.ndarray): array with shape (n_points,) of origin angles
                cone_angle (float): cone angle width
                target_pts (np.ndarray): target points to check whether in cones
            Returns:
                np.ndarray of bools. Each row corresponds to origin cone, and columns to
                    target points
        '''
        assert isinstance(origin_pts, np.ndarray)
        assert isinstance(origin_angles, np.ndarray)
        assert isinstance(cone_angle, float)
        assert isinstance(target_pts, np.ndarray)
        assert origin_pts.shape[0] == origin_angles.shape[0]
        assert len(origin_angles.shape) == 1, "Angles should only have 1 dimension"
        np.seterr(divide='ignore', invalid='ignore')
        cone_vec = np.array([np.cos(origin_angles), np.sin(origin_angles)]).T
        # Compute normed vectors between all pairs of agents
        pos_diffs = target_pts[None, ...] - origin_pts[:, None, :]
        norms = np.sqrt(np.sum(np.square(pos_diffs), -1, keepdims=True))
        unit_diffs = pos_diffs / norms
        # Dot product between unit vector in middle of cone and the vector
        dot_cone_diff = np.sum(unit_diffs * cone_vec[:, None, :], -1)
        angle_between = np.arccos(dot_cone_diff)
        # Right now the only thing that should be nan will be targets that are on the origin point
        # This can only happen for the origin looking at itself, so just make this always true
        angle_between[np.isnan(angle_between)] = 0.

        return np.abs(normalize_angles(angle_between)) <= cone_angle


# 观测范围内船舶的观测信息,这里只考虑了单船
class RadarObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_ships = self.unwrapped.n_ships
        self.n_reds = self.unwrapped.n_reds
        self.n_agents = self.unwrapped.n_agents

        self.red_ships = self.unwrapped.red_ships
        self.blue_ships = self.unwrapped.blue_ships


        self.features = ['distance','delta_x','delta_y','bearing_r','bearing_b','relative_u', 'relative_v', 'relative_r']
        self.unwrapped.observation_space = update_obs_space(env, {'radar_obs': (len(self.features),)})

    def reset(self,seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs),info


    def observation(self, obs):
        # obs['radar_obs'] = np.zeros((self.n_ships, 8))

        radar_obs = []
        # 红对蓝的探测
        for i, red_ship in enumerate(self.red_ships):
            for blue_ship in self.blue_ships:
                distance = distance_ss(red_ship, blue_ship)
                delta_x = blue_ship.pos[0] - red_ship.pos[0]
                delta_y = blue_ship.pos[1] - red_ship.pos[1]

                # 相对船首相的方位,随船坐标系，方位
                bearing_r = bearing_ss(red_ship, blue_ship)
                bearing_b = bearing_ss(blue_ship, red_ship)

                # relative_fi = red_ship.pos[2]- blue_ship.pos[2]
                # relative_fi = normalize_angles(relative_fi)

                u = blue_ship.vel[0]
                v = blue_ship.vel[1]
                r = blue_ship.vel[2]

            radar_obs.append([distance,delta_x,delta_y,bearing_r,bearing_b,
                                                u, v, r])

        # 蓝对红的探测
        for i, blue_ship in enumerate(self.blue_ships):
            for red_ship in self.red_ships:
                distance = distance_ss(red_ship, blue_ship)
                delta_x = red_ship.pos[0] - blue_ship.pos[0]
                delta_y = red_ship.pos[1] - blue_ship.pos[1]

                # 相对船首相的方位,随船坐标系，方位
                bearing_r = bearing_ss(blue_ship, red_ship)
                bearing_b = bearing_ss(red_ship, blue_ship)

                # relative_fi = red_ship.pos[2]- blue_ship.pos[2]
                # relative_fi = normalize_angles(relative_fi)

                u = red_ship.vel[0]
                v = red_ship.vel[1]
                r = red_ship.vel[2]

            radar_obs.append([distance,delta_x,delta_y,bearing_r,bearing_b,
                                                u, v, r])
            
        radar_obs = np.array(radar_obs).reshape(self.n_agents,-1)

        obs['radar_obs'] = radar_obs
        return obs
    

class OwnObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_ships = self.unwrapped.n_ships
        self.n_reds = self.unwrapped.n_reds
        self.n_agents = self.unwrapped.n_agents

        self.red_ships = self.unwrapped.red_ships
        self.blue_ships = self.unwrapped.blue_ships


        self.features = ['distance','delta_x','delta_y','bearing_r','bearing_b','relative_u', 'relative_v', 'relative_r']
        self.unwrapped.observation_space = update_obs_space(env, {'radar_obs': (len(self.features),)})

    def reset(self,seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs),info


    def observation(self, obs):
        # obs['radar_obs'] = np.zeros((self.n_ships, 8))

        # 红对蓝的探测
        own_obs = []
        for red_ship in self.red_ships:

            own_obs.append(np.hstack([red_ship.pos,red_ship.vel,red_ship.health]))

        for blue_ship in self.blue_ships:

            own_obs.append(np.hstack([blue_ship.pos,blue_ship.vel,blue_ship.health]))

        

        own_obs = np.array(own_obs).reshape(self.n_agents,-1)

        obs['ship_state'] = own_obs

        return obs


