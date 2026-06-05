import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.centraV import CentraV
import torch as th
from torch.optim import RMSprop
from torch import nn
import numpy as np
import torch.nn.functional as F



class MAPPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        
        self.mac = mac
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        # 共享的 Critic 网络
        self.critic = CentraV(scheme, args,self.n_agents)
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
        mask = mask.unsqueeze(2).expand(-1, -1, self.n_agents, -1)

        #只要一边的V值
        #没有分边，这里有个问题，如果是蓝船，输入的联合obs的顺序，也是按顺序排的，会有问题吗？

        with th.no_grad():
            target_v = self.critic(batch)
            advantages, returns = self._td_lambda_gae(
                rewards, terminated, critic_mask, target_v, self.n_agents, self.args.gamma, self.args.td_lambda
            )

            #这里虽然无效步的adv会不为0，但最后这里会被mask掉，所以问题不大。
            adv_mask = critic_mask.unsqueeze(-1)
            advantages = (advantages - advantages[adv_mask.bool()].mean()) / (
                advantages[adv_mask.bool()].std(unbiased=False) + 1e-8
            )
            advantages = advantages.expand(-1, -1, self.n_agents, -1)

            #policy
            actions = batch["actions"][:, :-1]
            self.mac.init_hidden(batch.batch_size)
            all_agent_outs = self.mac.forward(batch)
            mac_out = all_agent_outs[:, :-1]
            pi = mac_out
            pi_taken = th.gather(pi, dim=-1, index=actions)
            pi_taken = th.clamp(pi_taken, min=1e-8)
            #索引不支持广播
            pi_taken[mask == 0] = 1.0
            old_log_pi_taken = th.log(pi_taken).detach()

        # args.minibatch_size: when chunk_len is set, this becomes "chunks per minibatch"
        max_trans = max_t - 1  # number of transitions (actions/rewards)
        chunk_len = getattr(self.args, "chunk_len", None)
        if chunk_len is None or int(chunk_len) <= 0:
            chunk_len = max_trans
        else:
            chunk_len = int(chunk_len)
        burn_in = int(getattr(self.args, "burn_in", 0) or 0)
        burn_in = max(0, burn_in)

        # Build chunk index list: each entry is (episode_idx, chunk_start_t)
        # We use fixed-length chunks to keep tensors stackable: [t0 : t0+chunk_len] transitions
        # and [t0 : t0+chunk_len+1] timesteps for bootstrap.
        max_chunk_start = max_trans - chunk_len
        if max_chunk_start < 0:
            max_chunk_start = 0
        chunk_episode_ids = []
        chunk_start_ts = []
        for ep_i in range(bs):
            # If burn-in is enabled, only sample chunks with enough history.
            t0 = burn_in if burn_in > 0 else 0
            while t0 <= max_chunk_start:
                chunk_episode_ids.append(ep_i)
                chunk_start_ts.append(t0)
                t0 += chunk_len

        chunk_episode_ids = np.asarray(chunk_episode_ids, dtype=np.int64)
        chunk_start_ts = np.asarray(chunk_start_ts, dtype=np.int64)
        chunk_count = int(chunk_episode_ids.shape[0])
        if chunk_count == 0:
            chunk_episode_ids = np.arange(bs, dtype=np.int64)
            chunk_start_ts = np.zeros(bs, dtype=np.int64)
            chunk_count = int(chunk_episode_ids.shape[0])
        chunk_minibatch_size = int(getattr(self.args, "minibatch_size", 1) or 1)
        chunk_minibatch_size = max(1, chunk_minibatch_size)
        num_minibatches = max(1, int(np.ceil(chunk_count / chunk_minibatch_size)))

        def _gather_time(tensor, ep_ids, start_ts, length):
            """
            tensor: (bs, T, *rest)
            ep_ids: (n,)
            start_ts: (n,)
            returns: (n, length, *rest)
            """
            ep_ids_t = th.as_tensor(ep_ids, device=tensor.device, dtype=th.long)
            start_ts_t = th.as_tensor(start_ts, device=tensor.device, dtype=th.long)
            time_idx = start_ts_t[:, None] + th.arange(length, device=tensor.device, dtype=th.long)[None, :]

            selected = tensor.index_select(0, ep_ids_t)  # (n, T, *rest)
            rest_shape = selected.shape[2:]
            index = time_idx.view(time_idx.shape[0], time_idx.shape[1], *([1] * len(rest_shape))).expand(
                time_idx.shape[0], time_idx.shape[1], *rest_shape
            )
            return th.gather(selected, dim=1, index=index)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "actor_loss": [],
            "actor_grad_norm": [],
            "pi_max": [],
            "entropy_mean": [],
        }

        for _ in range(self.args.ppo_epochs): 
            chunk_perm = np.random.permutation(chunk_count)
            for i in range(num_minibatches):
                start_idx = i * chunk_minibatch_size
                end_idx = min(chunk_count, start_idx + chunk_minibatch_size)
                mb_chunk_idx = chunk_perm[start_idx:end_idx]
                mb_ep = chunk_episode_ids[mb_chunk_idx]
                mb_t0 = chunk_start_ts[mb_chunk_idx]

                # Build chunk batch with per-sample time windows
                chunk_batch = batch[mb_ep.tolist()]  # slice episodes first (supports duplicate eps)
                # Repack transition data to [n_chunks, chunk_len+1, ...]
                new_data = chunk_batch._new_data_sn()
                for k, v in chunk_batch.data.transition_data.items():
                    if v.shape[1] == max_t:
                        new_data.transition_data[k] = _gather_time(v, np.arange(len(mb_ep)), mb_t0, chunk_len + 1)
                    else:
                        # fields with T-1 transitions
                        new_data.transition_data[k] = _gather_time(v, np.arange(len(mb_ep)), mb_t0, chunk_len)
                for k, v in chunk_batch.data.episode_data.items():
                    new_data.episode_data[k] = v
                chunk_batch = EpisodeBatch(
                    chunk_batch.scheme,
                    chunk_batch.groups,
                    len(mb_ep),
                    chunk_len + 1,
                    preprocess=chunk_batch.preprocess,
                    data=new_data,
                    device=chunk_batch.device,
                )

                actions = chunk_batch["actions"][:, :-1]
                mb_old_log_pi = _gather_time(old_log_pi_taken, mb_ep, mb_t0, chunk_len).reshape(-1)
                mb_adv = _gather_time(advantages, mb_ep, mb_t0, chunk_len).reshape(-1)
                mb_mask = _gather_time(mask, mb_ep, mb_t0, chunk_len).reshape(-1)
                mb_returns = _gather_time(returns, mb_ep, mb_t0, chunk_len).reshape(-1)
                mb_critic_mask = _gather_time(critic_mask, mb_ep, mb_t0, chunk_len).reshape(-1)

                # Burn-in warmup on per-sample windows
                self.mac.init_hidden(chunk_batch.batch_size)
                if burn_in > 0:
                    burn_t0 = mb_t0 - burn_in
                    burn_batch = batch[mb_ep.tolist()]
                    burn_data = burn_batch._new_data_sn()
                    for k, v in burn_batch.data.transition_data.items():
                        burn_data.transition_data[k] = _gather_time(v, np.arange(len(mb_ep)), burn_t0, burn_in)
                    for k, v in burn_batch.data.episode_data.items():
                        burn_data.episode_data[k] = v
                    burn_batch = EpisodeBatch(
                        burn_batch.scheme,
                        burn_batch.groups,
                        len(mb_ep),
                        burn_in,
                        preprocess=burn_batch.preprocess,
                        data=burn_data,
                        device=burn_batch.device,
                    )
                    with th.no_grad():
                        self.mac.forward(burn_batch)

                all_agent_outs = self.mac.forward(chunk_batch)
                mac_out = all_agent_outs[:, :-1]
                pi = mac_out.reshape(-1, self.n_actions)
                pi_taken = th.gather(pi, dim=-1, index=actions.reshape(-1, 1)).squeeze(-1)
                pi_taken = th.clamp(pi_taken, min=1e-8)
                pi_taken[mb_mask == 0] = 1.0
                new_log_pi_taken = th.log(pi_taken)

                log_ratio = new_log_pi_taken - mb_old_log_pi
                # Prevent exp overflow (inf) when log-ratio becomes too large.
                max_log_ratio = float(getattr(self.args, "max_log_ratio", 5.0) or 5.0)
                log_ratio = th.clamp(log_ratio, -max_log_ratio, max_log_ratio)
                ratio = th.exp(log_ratio)
                surr1 = ratio * mb_adv
                surr2 = th.clamp(ratio, 1 - self.args.clip_range, 1 + self.args.clip_range) * mb_adv
                surr = th.min(surr1, surr2)
                actor_loss = - (surr * mb_mask).sum() / (mb_mask.sum() + 1e-8)

                eps = 1e-10
                log_pi = th.log(pi + eps)
                mb_entropy = -(pi * log_pi).sum(dim=-1)
                entropy_loss = (mb_entropy * mb_mask).sum() / (mb_mask.sum() + 1e-8)
                loss = actor_loss - self.args.ent_coef * entropy_loss

                new_values = self.critic(chunk_batch)[:, :-1].reshape(-1)
                error = mb_returns.detach() - new_values
                critic_loss = ((error ** 2) * mb_critic_mask).sum() / (mb_critic_mask.sum() + 1e-8)

                self.agent_optimiser.zero_grad()
                loss.backward()
                actor_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
                self.agent_optimiser.step()

                self.critic_optimiser.zero_grad()
                critic_loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
                self.critic_optimiser.step()
                self.critic_training_steps += 1

                if _ == 0:
                    running_log["critic_loss"].append(critic_loss.item())
                    running_log["actor_loss"].append(actor_loss.item())
                    running_log["critic_grad_norm"].append(grad_norm.item())
                    running_log["actor_grad_norm"].append(actor_grad_norm.item())
                    running_log["pi_max"].append(
                        (pi.max(dim=1)[0][mb_mask > 0].min().item())
                    )
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
