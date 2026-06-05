from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from components.episode_buffer import ReplayBuffer
import numpy as np
import torch as th


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args[0]
        self.args2 = args[1]
        self.n_reds = self.args.env_args["n_reds"]
        self.logger = logger
        self.batch_size = 1
        assert self.batch_size == 1

        #这里直接输出unwrapped，就可以得到最原始的env。不可以这么做，这样做，reset这个env的时候，就不会包装了。
        self.env = env_REGISTRY[self.args.env](args_dict=self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, red_mac,blue_mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac1 = red_mac
        self.mac2 = blue_mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self,options=None):
        self.batch = self.new_batch()
        obs,info = self.env.reset(options=options)
        self.t = 0
        return obs,info

    def run(self, test_mode=False,train_callback=None,options=None):
        obs,info = self.reset(options=options)

        terminated = False
        truncated = False
        
        episode_return = 0
        self.mac1.init_hidden(batch_size=self.batch_size)
        self.mac2.init_hidden(batch_size=self.batch_size)

        while not (terminated or truncated):         
            #进入神经网络,虽然规则策略用不到blue_obs，但是，red的训练会用到。
            flatten_obs = self.env.get_flatten_obs(obs)

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": flatten_obs,
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions1 = self.mac1.select_actions(self.batch[:,:,:self.n_reds], t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            #不能用红方的mac处理蓝方的数据，即使同质也不行，因为涉及到隐藏层，
            actions2 = self.mac2.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        
            actions1_env = actions1.detach().cpu().view(-1).numpy()
            actions2_env = actions2.detach().cpu().view(-1).numpy()
            joint_actions_env = list([actions1_env, actions2_env])
            joint_actions_np = np.concatenate([actions1_env, actions2_env])

            obs,reward, terminated, truncated,info = self.env.step(joint_actions_env)
            #主流RL的指标都是不折扣的奖励
            episode_return += reward

            post_transition_data = {
                "actions": joint_actions_np,
                "reward": [(reward,)],
                "terminated": [(terminated,)],
            }


            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        #进入神经网络
        flatten_obs = self.env.get_flatten_obs(obs)

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": flatten_obs
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions1 = self.mac1.select_actions(self.batch[:,:,:self.n_reds], t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

        #不能用红方的mac处理蓝方的数据，即使同质也不行，因为涉及到隐藏层，
        actions2 = self.mac2.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        
        actions1_env = actions1.detach().cpu().view(-1).numpy()
        actions2_env = actions2.detach().cpu().view(-1).numpy()
        joint_actions_np = np.concatenate([actions1_env, actions2_env])

        self.batch.update({"actions": joint_actions_np}, ts=self.t)

        env_info = self.env.get_env()

        cur_stats = self.test_stats if options else self.train_stats
        cur_returns = self.test_returns if options else self.train_returns
        log_prefix = "test_" if options else ""
        #这里会记录env_info 的参数,这里会叠加，win的次数会叠加
        cur_stats.update({k: cur_stats.get(k, []) + [env_info.get(k, [])] for k in set(env_info)})
        # cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        # #red—win次数
        # cur_stats["red_win"] = self.t + cur_stats.get("ep_length", 0)

        if not options:
            self.t_env += self.t
        #这里是记录每一次的returns，记录每一次的可以类似这个
        cur_returns.append(episode_return)

        if options and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif not options and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac1.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac1.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns,axis=0), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns,axis=0), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k == "health":
                v= np.array(v)
                self.logger.log_stat(prefix + k + "_mean" , np.mean(v,axis=0), self.t_env)
                self.logger.log_stat(prefix + k + "_std", np.std(v,axis=0), self.t_env)
                self.logger.log_stat(prefix + k + "_min", np.min(v,axis=0), self.t_env)
                score = v[:,0]-v[:,1]
                self.logger.log_stat(prefix + "score" + "_mean" , np.mean(score,axis=0), self.t_env)
                self.logger.log_stat(prefix + "score" + "_std", np.std(score,axis=0), self.t_env)
                self.logger.log_stat(prefix + "score" + "_min", np.min(score,axis=0), self.t_env)
            if k =="ep_length":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
            if k == "game_result":
                self.logger.log_stat(prefix + "win_rate", v.count(1)/stats["n_episodes"], self.t_env)
                self.logger.log_stat(prefix + "loss_rate", v.count(-1)/stats["n_episodes"], self.t_env)
        stats.clear()
