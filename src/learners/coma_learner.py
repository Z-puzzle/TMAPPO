import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop


class COMALearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_actions = args.n_actions
        self.mac = mac
        self.side = self.mac.side
        self.n_reds = args.n_reds
        self.n_blues = args.n_blues
        self.logger = logger
        if self.side == "red":
            self.n_agents = args.n_reds
        else:
            self.n_agents = args.n_blues

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = COMACritic(scheme, args,self.side)
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        if self.side =="red":
            #actions1 代表自己这一边，acitons2 代表敌人那一边
            actions = batch["actions"][:, :,:self.n_reds]
            #把group的维度去掉，设计的是两边的reward。所以要去掉智能体维度。
            rewards = batch["reward"][:, :-1,:1].squeeze(2)
        else:
            actions = batch["actions"][:, :,self.n_reds:]
            rewards = batch["reward"][:, :-1,1:].squeeze(2)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        critic_mask = mask.clone()

        mask = mask.repeat(1, 1, self.n_agents).view(-1)

        q_vals, critic_train_stats = self._train_critic(batch, rewards, terminated, actions, avail_actions,
                                                        critic_mask, bs, max_t)
        
        target_v = self.target_critic(batch)[:, :,:self.n_reds]
        targets_taken = th.gather(target_v, dim=3, index=actions).squeeze(3)
        returns = build_td_lambda_targets(rewards, terminated, critic_mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda)
        returns = returns.reshape(-1)

        actions = actions[:,:-1]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.cat(mac_out, dim=1)  # Concat over time，每个时间的动作输出

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        # Calculated baseline
        q_vals = q_vals.reshape(-1, self.n_actions)
        pi = mac_out.view(-1, self.n_actions)
        baseline = (pi * q_vals).sum(-1).detach()

        # Calculate policy grad with mask
        q_taken = th.gather(q_vals, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)

        log_pi_taken.retain_grad()

        # advantages = (returns - baseline).detach()
        advantages = (q_taken - baseline).detach()
        advantages = (advantages - advantages[mask.bool()].mean()) / (advantages[mask.bool()].std(unbiased=False)+1e-8)

        coma_loss1 = - ((advantages * log_pi_taken) * mask).sum() / mask.sum()

        #熵正则项
        eps = 1e-10

        log_pi = th.log(pi+eps)
        entropy = -(pi * log_pi).sum(dim=-1)    # [bs*T*n_agents]
        entropy_mean = (entropy * mask).sum() / mask.sum() #entropy_mean 表示当前整个 batch 的平均策略熵，反映策略整体的“分散程度”。

        coma_loss = coma_loss1 - self.args.ent_coef * entropy_mean

        # Optimise agents
        self.agent_optimiser.zero_grad()
        coma_loss.backward()

        #测试每个动作的优势
        g = log_pi_taken.grad.reshape(-1)
        a = actions.reshape(-1)
        for act in sorted(a.unique().tolist()):
            mask_t = (a == act) & (mask ==1)
            print(f"act={act}: mean_grad={g[mask_t].mean().item():.4e}, count={mask_t.sum().item()}")        

        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        # if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
        #     self._update_targets()
        #     self.last_target_update_step = self.critic_training_steps

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key])/ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("coma_loss", coma_loss.item(), t_env)
            self.logger.log_stat("entropy_mean", entropy_mean.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def _train_critic(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t):
        # Optimise critic
        #用一边的奖励训练Q网络，这不影响外面的actor loss 计算
        if self.side == "red":
            target_q_vals = self.target_critic(batch)
        else:
            target_q_vals = self.target_critic(batch)
        targets_taken = th.gather(target_q_vals, dim=3, index=actions).squeeze(3)

        # Calculate td-lambda targets
        targets = build_td_lambda_targets(rewards, terminated, mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda)

        q_vals = th.zeros_like(target_q_vals)[:, :-1]

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        for t in reversed(range(rewards.size(1))):
            mask_t = mask[:, t].expand(-1, self.n_agents)
            if mask_t.sum() == 0:
                continue

            if self.side == "red":
                q_t = self.critic(batch,t)[:, :,:self.n_reds]
            else:            
                q_t = self.critic(batch,t)[:, :,self.n_reds:]
            q_vals[:, t] = q_t.view(bs, self.n_agents, self.n_actions)
            q_taken = th.gather(q_t, dim=3, index=actions[:, t:t+1]).squeeze(3).squeeze(1)
            targets_t = targets[:, t]

            td_error = (q_taken - targets_t.detach())

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask_t

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask_t.sum()
            self.critic_optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            self.critic_training_steps += 1
            self._update_targets()

            running_log["critic_loss"].append(loss.item())
            running_log["critic_grad_norm"].append(grad_norm.item())
            mask_elems = mask_t.sum().item()
            running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
            running_log["q_taken_mean"].append((q_taken * mask_t).sum().item() / mask_elems)
            running_log["target_mean"].append((targets_t * mask_t).sum().item() / mask_elems)

        return target_q_vals[:, :-1], running_log

    def _update_targets(self):
        tau=0.01
        with th.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.mul_(1 - tau)
                target_param.data.add_(tau * param.data)

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
