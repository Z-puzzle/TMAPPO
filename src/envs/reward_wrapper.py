import gymnasium as gym
import numpy as np
from .utils import *


class TerminatedWrapper(gym.Wrapper):
    def __init__(self, env, reward_scale=1.0):
        super().__init__(env)
        self.n_reds = self.unwrapped.n_reds
        self.n_blues = self.unwrapped.n_blues
        self.n_ships = self.unwrapped.n_ships
        self.ships = self.unwrapped.ships
        self.reward_scale = reward_scale
        self.status = None

        self.episode_limit = self.unwrapped.episode_limit

    def reset(self, *, seed = None, options = None):
        self.status = None
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)

        this_rew = np.zeros(2)

        health = np.array([ship.health for ship in self.ships])

        red_dead = np.all(health[:self.n_reds] <= 0)  # 红队所有飞船健康值 <= 0
        blue_dead = np.all(health[self.n_reds:] <= 0)  # 蓝队所有飞船健康值 <= 0

        self.winner(red_dead, blue_dead)

        if truncated:
            self.status = 2

        # # 超时惩罚，terminated不能为true
        if self.status == 2:
            this_rew[0] = -20 + (np.mean(health[:self.n_reds]) - np.mean(health[self.n_reds:]))
            this_rew[1] = -20 + -(np.mean(health[:self.n_reds]) - np.mean(health[self.n_reds:]))
        
        # # 不是超时，终局奖励，平方不能体现正负性，立方又太大了
        if self.status ==1 or self.status ==-1 or self.status == 0:
            this_rew[0] = (np.mean(health[:self.n_reds]) - np.mean(health[self.n_reds:]))
            this_rew[1] = -(np.mean(health[:self.n_reds]) - np.mean(health[self.n_reds:]))
            terminated = True

        rew += this_rew*self.reward_scale

        return obs, rew, terminated, truncated, info
    
    def get_state(self):
        return self.unwrapped.get_state()

    def winner(self, red_dead, blue_dead):

        if red_dead and blue_dead:
            self.status = 0  # 如果两个条件都成立
        # 红船赢
        elif blue_dead:
            self.status = 1
        # 蓝船赢
        elif red_dead:
            self.status = -1

    def get_avail_actions(self):
        return self.unwrapped.get_avail_actions()
    
    def get_env(self):
        env_info = {"health": np.array([self.ships[0].health,self.ships[1].health]),

                    "game_result": self.status,
                    }
        return env_info

    def get_flatten_obs(self, obs):
        return self.unwrapped.get_flatten_obs(obs)
    
    def get_env_info(self):
        return self.unwrapped.get_env_info()

    def save_replay(self):
        return self.unwrapped.save_replay()
    

# health 经过运动生命值变化
class HealthRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_ships = self.unwrapped.n_ships
        self.n_reds = self.unwrapped.n_reds
        self.n_blues = self.unwrapped.n_blues
        self.ships = self.unwrapped.ships

    def step(self, actions):
        obs, rew, terminated, truncated, info = self.env.step(actions)
        this_rew = np.zeros(2)

        red_delta_health = 0.0
        blue_delta_health = 0.0

        for i, ship in enumerate(self.ships[:self.n_reds]):
            red_delta_health += ship.prev_health - ship.health
        for i, ship in enumerate(self.ships[self.n_reds:]):
            blue_delta_health += ship.prev_health - ship.health

        #这里有时间步惩罚-1,这里只分边了，因为计算的是总的生命值损失
        this_rew[0] = blue_delta_health - red_delta_health -0.05
        this_rew[1] = red_delta_health - blue_delta_health -0.05

        rew += this_rew

        return obs, rew, terminated, truncated, info

class OutRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_ships = self.env.unwrapped.n_ships

    def step(self, actions):
        obs, rew, terminated, truncated, info = self.env.step(actions)
        this_rew = np.zeros(self.n_ships)
        for i in range(self.n_ships):
            if np.any(abs(self.ships[i].pos[:2]) >= 300):
                this_rew[i] -= 10.0

        rew += this_rew

        return obs, rew, terminated, truncated, info

class NavigateWrapper(gym.Wrapper):
    def __init__(self, env, reward_scale=1.0):
        super().__init__(env)
        self.n_reds = self.unwrapped.n_reds
        self.n_blues = self.unwrapped.n_blues
        self.n_ships = self.unwrapped.n_ships

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        this_rew = np.zeros(self.n_ships)

        this_rew[:self.n_reds] = -1.0

        rew += this_rew

        return obs, rew, terminated, truncated, info

class CollisionWrapper(gym.Wrapper):
    def __init__(self, env, reward_scale=1.0):
        super().__init__(env)
        self.n_reds = self.unwrapped.n_reds
        self.n_blues = self.unwrapped.n_blues
        self.n_ships = self.unwrapped.n_ships
        self.reward_scale = reward_scale

        self.alpha = 1.0
        #碰撞区域，小于rmax风险为1
        self.r_max = 3
    def reset(self,seed=None,options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs,info
    def caculate_field(self, x, y, x0, y0):
        '''
        :param x: x，y位置的场强
        :param y:
        :param fi0: 形成场的船的角度
        :param x0: x0，y0，x0，y0形成的场
        :param y0:
        :return:
        '''
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)  # 距离

        # return np.exp(-self.alpha*(r-20))*np.exp(-self.beta*(theta-3/8*np.pi))
        return np.exp(-self.alpha*(r-self.r_max))

    def normolized_filed(self,E_c):
        if E_c >= 1.0:
            E_c=1.0
        else:
            E_c=E_c/1.0

        return E_c

    def step(self,action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        this_rew = np.zeros(self.n_ships)

        red_ship_pos,blue_ship_pos = obs['ship_pos'][0], obs['ship_pos'][1]

        #蓝船对红船
        E_RinB = self.caculate_field(red_ship_pos[0], red_ship_pos[1], blue_ship_pos[0],blue_ship_pos[1])
        E_RinB = self.normolized_filed(E_RinB)
        # 红船对蓝船
        E_BinR = self.caculate_field(blue_ship_pos[0],blue_ship_pos[1],red_ship_pos[0],red_ship_pos[1])
        E_BinR = self.normolized_filed(E_BinR)

        reward3 = -E_RinB
        #
        this_rew[:self.n_reds] = reward3

        rew += this_rew *self.reward_scale
        return obs, rew, terminated, truncated, info


class PBRSRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_scale=1.0):
        super().__init__(env)
        self.n_reds = self.unwrapped.n_reds
        self.n_blues = self.unwrapped.n_blues
        self.n_ships = self.unwrapped.n_ships
        self.reward_scale = reward_scale
        self.args = self.unwrapped.args
        self.gamma = self.args["gamma"]

        self.r_max = 20
        self.angel_max = self.unwrapped.ships[0].fire_angle
        #risk,代表2，10 对应0。1，1. ；2，5对应 0。1，2
        self.alpha = np.log(9)/((1/2)*20*(self.args['alpha']))
        self.beta = np.log(9)/((1/2)*(1/8*np.pi)*(self.args['beta']))
        # #base，模拟health扣除的风险势场
        # self.alpha = 10
        # self.beta = 50

        self.max_value_infield = self.caculate_field(0,0,0,0,0)

    def reset(self,seed=None,options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        red_ship_pos = obs['ship_state'][0,:3]
        blue_ship_pos = obs['ship_state'][1,:3]

        #蓝船对红船，敌船对己船
        E_RinB = self.caculate_field(red_ship_pos[0], red_ship_pos[1], blue_ship_pos[0],blue_ship_pos[1],blue_ship_pos[2])
        E_RinB = self.normolized_filed(E_RinB)
        # 红船对蓝船，红打击蓝，己船对敌船
        E_BinR = self.caculate_field(blue_ship_pos[0],blue_ship_pos[1],red_ship_pos[0],red_ship_pos[1],red_ship_pos[2])
        E_BinR = self.normolized_filed(E_BinR)

        fi = 0.5*E_BinR + 0.5 *(-E_RinB)

        self.old_fi = fi

        return obs,info

    def caculate_field(self, x, y, x0, y0,fi0):
        '''
        :param x: x，y位置的场强
        :param y:
        :param fi0: 形成场的船的角度
        :param x0: x0，y0，x0，y0形成的场
        :param y0:
        :return:
        '''
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)  # 距离
        angle = np.arctan2((y - y0), (x - x0))
        bearing = angle if angle >= 0 else angle + 2 * np.pi

        theta = normalize_angles(bearing - fi0)
        theta = abs(theta)

        # return np.exp(-self.alpha*(r-20))*np.exp(-self.beta*(theta-3/8*np.pi))
        return (1 / (1 + np.exp(self.alpha * (r - 20)))) * (1 / (1 + np.exp(self.beta * (theta - self.angel_max))))
    def normolized_filed(self,E_value):
        E_value = E_value/self.max_value_infield
        return E_value

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        this_rew = np.zeros(self.n_ships)

        red_ship_pos = obs['ship_state'][0,:3]
        blue_ship_pos = obs['ship_state'][1,:3]

        #蓝船对红船，敌船对己船
        E_RinB = self.caculate_field(red_ship_pos[0], red_ship_pos[1], blue_ship_pos[0],blue_ship_pos[1],blue_ship_pos[2])
        E_RinB = self.normolized_filed(E_RinB)
        # 红船对蓝船，红打击蓝，己船对敌船
        E_BinR = self.caculate_field(blue_ship_pos[0],blue_ship_pos[1],red_ship_pos[0],red_ship_pos[1],red_ship_pos[2])
        E_BinR = self.normolized_filed(E_BinR)


        fi = 0.5*E_BinR + 0.5 *(-E_RinB)

        #终局的fi要为0，才满足定义。
        if terminated and not truncated:
            fi = 0      

        # risk,
        this_rew[:self.n_reds] = self.gamma*fi - self.old_fi

        rew += this_rew *self.reward_scale
        return obs, rew, terminated, truncated, info


class FieldRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_scale=1.0):
        super().__init__(env)
        self.n_reds = self.unwrapped.n_reds
        self.n_blues = self.unwrapped.n_blues
        self.n_ships = self.unwrapped.n_ships
        self.reward_scale = reward_scale
        self.args = self.unwrapped.args
        self.r1 = self.args['r1']
        self.r2 = self.args['r2']
        self.r3 = self.args['r3']

        self.r_max = 20
        self.angel_max = self.unwrapped.ships[0].fire_angle
        #risk,代表2，10 对应0。1，1. ；2，5对应 0。1，2
        self.alpha = np.log(9)/((1/2)*20*(self.args['alpha']))
        self.beta = np.log(9)/((1/2)*(1/8*np.pi)*(self.args['beta']))
        # #base，模拟health扣除的风险势场
        # self.alpha = 10
        # self.beta = 50

        self.max_value_infield = self.caculate_field(0,0,0,0,0)

    def reset(self,seed=None,options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.old_E_BinR = None
        self.old_E_RinB = None
        return obs,info

    def caculate_field(self, x, y, x0, y0,fi0):
        '''
        :param x: x，y位置的场强
        :param y:
        :param fi0: 形成场的船的角度
        :param x0: x0，y0，x0，y0形成的场
        :param y0:
        :return:
        '''
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)  # 距离
        angle = np.arctan2((y - y0), (x - x0))
        bearing = angle if angle >= 0 else angle + 2 * np.pi

        theta = normalize_angles(bearing - fi0)
        theta = abs(theta)

        # return np.exp(-self.alpha*(r-20))*np.exp(-self.beta*(theta-3/8*np.pi))
        return (1 / (1 + np.exp(self.alpha * (r - 20)))) * (1 / (1 + np.exp(self.beta * (theta - self.angel_max))))
    def normolized_filed(self,E_value):
        E_value = E_value/self.max_value_infield
        return E_value

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        this_rew = np.zeros(self.n_ships)

        red_ship_pos = obs[0]['ship_state'][:3]
        blue_ship_pos = obs[1]['ship_state'][:3]

        #蓝船对红船
        E_RinB = self.caculate_field(red_ship_pos[0], red_ship_pos[1], blue_ship_pos[0],blue_ship_pos[1],blue_ship_pos[2])
        E_RinB = self.normolized_filed(E_RinB)
        # 红船对蓝船，红打击蓝
        E_BinR = self.caculate_field(blue_ship_pos[0],blue_ship_pos[1],red_ship_pos[0],red_ship_pos[1],red_ship_pos[2])
        E_BinR = self.normolized_filed(E_BinR)

        # 如果是第一回合（old_E_BinR 和 old_E_RinB 尚未定义）
        # if not hasattr(self, "old_E_BinR") or not hasattr(self, "old_E_RinB"):
        if self.old_E_BinR is None or self.old_E_RinB is None:
            self.old_E_RinB = E_RinB
            self.old_E_BinR = E_BinR

        # 红船奖励，对敌方造成的风险，比敌方对我方造成的风险多，为正。态势变化风险。
        # delta_E_RinB = E_RinB - self.old_E_RinB
        # delta_E_BinR = E_BinR - self.old_E_BinR
        #
        # this_rew[:self.n_reds] = delta_E_BinR - delta_E_RinB
        # # this_rew[:self.n_reds] = -E_RinB

        # 态势奖励，作战态势好为正，红船奖励，对敌方造成的风险，比敌方对我方造成的风险多，为正。态势风险差
        reward1 = E_BinR - E_RinB
        reward2 = self.old_E_RinB - E_RinB
        #危险度
        reward3 = -E_RinB

        # risk,
        this_rew[:self.n_reds] = self.r1*reward1 + self.r2*reward2 + self.r3*reward3
        # base，是只有1.0，reward1，-1是超时的惩罚，每一步都有个时间惩罚
        # this_rew[:self.n_reds] = 1 * reward1 + 0.0 * reward2 + 0.0 * reward3 - 1.0

        dis = distance_ss(self.unwrapped.ships[0],self.unwrapped.ships[1])

        self.old_E_RinB = E_RinB
        self.old_E_BinR = E_BinR

        rew += this_rew *self.reward_scale
        return obs, rew, terminated, truncated, info