import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.independV import IndependV
import torch as th
from torch.optim import RMSprop
from torch import nn
import numpy as np
import torch.nn.functional as F



class IPPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        
        self.mac = mac
        #单边的算法需要分边，输入的obs不同，
        self.n_reds = args.n_reds
        self.n_blues = args.n_blues

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger
        

        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        # 共享的 Critic 网络
        self.critic = IndependV(scheme, args,self.n_agents)

        self.target_critic = copy.deepcopy(self.critic)


        # 参数列表，共享的网络，只有单个参数
        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())  # 单个 Critic 的参数列表

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        #ppo 不加lstm。原来的代码我没加。
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]

        terminated = batch["terminated"][:, :-1].float()
        #fill是1，说明这是一个真实时间步
        #mask是1，说明这个时间步有效
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        
        critic_mask = mask.clone()

        mask = mask.unsqueeze(2).expand(-1,-1,self.n_agents,-1)


        #policy
        with th.no_grad():
            target_v = self.critic(batch)


            advantages, returns = self._td_lambda_gae(rewards, terminated, critic_mask, target_v, self.n_agents, self.args.gamma, self.args.td_lambda)

            #这里虽然无效步的adv会不为0，但最后这里会被mask掉，所以问题不大。
            advantages = (advantages - advantages[mask.bool()].mean()) / (advantages[mask.bool()].std(unbiased=False)+1e-8)
            # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            # advantages = advantages*5

            actions = batch["actions"][:, :-1]

            self.mac.init_hidden(batch.batch_size)
            # 2. 一次性算出全序列 (结果形状为 [B, max_t, A, N])
            all_agent_outs = self.mac.forward(batch)
            # 3. 切片取得前 max_t - 1 步，逻辑上完全等价于你的循环结果
            mac_out = all_agent_outs[:, :-1]         
            pi = mac_out
            pi_taken = th.gather(pi, dim=-1, index=actions)
            pi_taken = th.clamp(pi_taken, min=1e-8)
            #索引不支持广播
            pi_taken[mask == 0] = 1.0
            old_log_pi_taken = th.log(pi_taken).detach()

        # 生成 Episode 索引 [0, 1, ..., bs-1]
        episode_indices = np.arange(bs) 
        # args.minibatch_size 这里指包含多少条 Episode (建议设为 4 或 8)
        num_minibatches = max(1, bs // self.args.minibatch_size)               


        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "actor_loss": [],
            "actor_grad_norm": [],
            "pi_max": [],
            "entropy_mean": [],
        }

        for _ in range(self.args.ppo_epochs): 
            np.random.shuffle(episode_indices)
            for i in range(num_minibatches):
                # 提取 minibatch 的索引
                start_idx = i * self.args.minibatch_size
                end_idx = start_idx + self.args.minibatch_size
                if i == num_minibatches - 1:
                    end_idx = bs
                mb_idx = episode_indices[start_idx:end_idx]

                mb_batch = batch[mb_idx]
                mb_old_log_pi = old_log_pi_taken[mb_idx].view(-1)
                mb_adv = advantages[mb_idx].view(-1)
                mb_mask = mask[mb_idx].view(-1)
                mb_returns = returns[mb_idx].view(-1)

                actions = mb_batch["actions"][:, :-1]

                self.mac.init_hidden(mb_batch.batch_size)
                # 2. 一次性算出全序列 (结果形状为 [B, max_t, A, N])
                all_agent_outs = self.mac.forward(mb_batch)
                # 3. 切片取得前 max_t - 1 步，逻辑上完全等价于你的循环结果
                mac_out = all_agent_outs[:, :-1]
                pi = mac_out.reshape(-1,self.n_actions)
                pi_taken = th.gather(pi, dim=-1, index=actions.reshape(-1,1)).squeeze(-1)
                pi_taken = th.clamp(pi_taken, min=1e-8)
                #索引不支持广播
                pi_taken[mb_mask == 0] = 1.0
                new_log_pi_taken = th.log(pi_taken)


                ratio = th.exp(new_log_pi_taken - mb_old_log_pi)

                #advantages 在计算时已经mask了。
                # 近端策略优化裁剪目标函数公式的左侧项
                surr1 = ratio * mb_adv
                # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
                surr2 = th.clamp(ratio, 1 - self.args.clip_range, 1 + self.args.clip_range) * mb_adv

                # 策略网络的损失函数
                surr = th.min(surr1, surr2)
                actor_loss =  - (surr * mb_mask).sum() / (mb_mask.sum() + 1e-8)


                #entropy loss
                eps = 1e-10

                #避免pi很小时，log出现inf，加一个eps不要紧的，但不要截断，截断会丢失梯度。
                log_pi = th.log(pi+eps)
                mb_entropy = -(pi * log_pi).sum(dim=-1)                     
                entropy_loss = (mb_entropy * mb_mask).sum() / mb_mask.sum()

                # 总损失
                loss = actor_loss - self.args.ent_coef * entropy_loss



                #价值网络
                
                new_values = self.critic(mb_batch)[:, :-1].reshape(-1)
                
                error = mb_returns.detach() - new_values

                critic_loss = ((error ** 2) * mb_mask).sum() / mb_mask.sum()


                self.agent_optimiser.zero_grad()
                loss.backward()
                actor_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
                self.agent_optimiser.step()

                self.critic_optimiser.zero_grad()
                critic_loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
                self.critic_optimiser.step()
                self.critic_training_steps += 1
                #一般不用
                # self._update_targets()

                if _==0:
                    running_log["critic_loss"].append(critic_loss.item())
                    running_log["actor_loss"].append(actor_loss.item())
                    running_log["critic_grad_norm"].append(grad_norm.item())
                    running_log["actor_grad_norm"].append(actor_grad_norm.item())
                    # running_log["pi_max"].append((pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item())
                    running_log["entropy_mean"].append(entropy_loss.item())

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(running_log["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "actor_loss", "actor_grad_norm", "pi_max","entropy_mean"]:
                self.logger.log_stat(key, sum(running_log[key])/ts_logged, t_env)
            self.log_stats_t = t_env
        
    def _update_targets(self):
        tau=0.01
        with th.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.mul_(1 - tau)
                target_param.data.add_(tau * param.data)

    def _td_lambda_gae(self,rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
        # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
        # Initialise  last  lambda -return  for  not  terminated  episodes
        mask = mask.unsqueeze(-1)
        values = target_qs[:,:-1]
        next_values = target_qs[:,1:]

        td_delta = rewards + gamma * next_values * (1 - terminated.unsqueeze(2)) - values

        advantages = th.zeros_like(td_delta)
        gae_returns = th.zeros_like(td_delta)

        #这里有问题，没搞到255，最后一步adv永远是0，修改了for循环-1.
        last_gae_adv = 0
        for t in range(td_delta.shape[1] - 1, -1,  -1):
            mask_t = mask[:, t]
            current_gae_adv = td_delta[:, t] + gamma * td_lambda * last_gae_adv

            # 应用 mask: 如果该时间步无效，优势也应该是0
            current_gae_adv = current_gae_adv * mask_t

            last_gae_adv = current_gae_adv

            advantages[:, t] = current_gae_adv

        gae_returns = advantages + values

        return advantages,gae_returns

    def soft_update(self,target, source, t):
        for target_param, source_param in zip(target.parameters(),
                                            source.parameters()):
            target_param.data.copy_(
                (1 - t) * target_param.data + t * source_param.data)

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        # 收集所有优化器的 state_dict
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
    def load_mac(self, path):
        self.mac.load_models(path)
        # self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks

