import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop
from torch import nn
import numpy as np
import torch.nn.functional as F



class RuleLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        
        self.mac = mac
        #单边的算法需要分边，输入的obs不同，
        self.n_reds = args.n_reds
        self.n_blues = args.n_blues
        self.n_agents = args.n_agents

        self.n_actions = args.n_actions
        self.logger = logger

    def train(self,batch: EpisodeBatch, t_env: int, episode_num: int):
        pass
    def cuda(self):
        pass

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass
