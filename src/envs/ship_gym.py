from .physics.dynamics import ShipMotion
import math
import numpy as np
from collections import deque


class Ship():
    def __init__(self,name,args):
        
        self.name = name
        self.args= args

        self.radar_range = self.args['radar_range']  # 雷达观测范围
        self.fire_range = self.args['fire_range']  # 打击距离
        self.fire_angle = self.args['fire_angle']  # 打击范围
        
        # 初始化模块
        self.ship_motion = ShipMotion()  # 运动模块
        self.typeid=None

        self.reset()

    def reset(self):
        self.health = self.args['max_health']
        self.health = self.args['max_health']  # 当前生命值
        self.prev_health = self.args['max_health']    # 上一帧生命值
        self.pos = np.array([0., 0., 0.])  # [x, y, theta]
        self.vel = np.array([0., 0., 0.]) # 地坐标系下[vx, vy, r]

        self.ship_motion.reset()
        self.generate_position()

    def update_ship_state(self):
        self.pos = np.copy(self.ship_motion._state)
        self.vel = np.copy(self.ship_motion._global_vel)

    def update_health(self, new_health):
        self.prev_health = self.health  # 更新前保存当前生命值
        self.health = max(0, min(new_health, self.args['max_health']))       # 更新当前生命值

    def generate_position(self):
        """在World范围内随机生成位置和角度"""
        x = np.random.uniform(-100.0, 100.0)
        y = np.random.uniform(-100.0, 100.0)
        theta = np.random.uniform(0, 2 * math.pi)  # 随机生成弧度制的角度
        state = np.array([x, y, theta])

        self.set_position(state)

        return state

    def set_position(self, state):
        """设定船舶的位置和角度"""
        self.ship_motion._state = np.array(state)

        self.update_ship_state()

    def move(self, tw=None, tx=None, u =None):
        """调用ShipMotion模块的move接口，同步更新船舶的状态"""
        # 根据输入舵角和转速，移动船舶当前位置,
        # tw有上限即扭矩有上限。舵角可以近似等同于扭矩。舵角越大，扭矩越大。
        # 实际情况，舵角不能从一个最大值到一个最小值。这里也应该要加限制。
        self.ship_motion.move(tw, tx, u) 

        self.update_ship_state()