from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from torch import nn

class CommNetMAC:
    def __init__(self, scheme, groups, args, side):
        self.args = args
        self.side = side
        self.n_reds = args.n_reds
        self.n_blues = args.n_blues
        
        # 1. 区分红蓝方确定智能体数量
        if side == "red":
            self.n_agents = args.n_reds
        else:
            self.n_agents = args.n_blues        

        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        ts = slice(None) if t_ep is None else slice(t_ep, t_ep+1)
        avail_actions = ep_batch["avail_actions"][:, ts]
        
        # Forward 会处理 hidden states 的更新
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t=None, test_mode=False):
        # 1. 构建输入
        ts = slice(None) if t is None else slice(t, t+1)
        max_t = ep_batch.max_seq_length if t is None else 1
        
        # agent_inputs shape: [bs * max_t * n_agents, input_dim]
        agent_inputs = self._build_inputs(ep_batch, max_t, ts)
        avail_actions = ep_batch["avail_actions"][:, ts]
        
        bs = ep_batch.batch_size
        
        # 2. Reshape 以便进行按时间步的处理和通信聚合
        # [bs * max_t * n_agents, dim] -> [bs, max_t, n_agents, dim]
        agent_inputs = agent_inputs.reshape(bs, max_t, self.n_agents, -1)
        
        # 获取当前的 hidden_state [bs, n_agents, hidden_dim]
        # 注意：如果是训练模式(t=None)，这里的hidden_states应该是初始状态
        # 如果是执行模式(t!=None)，这里是上一步的hidden_state
        if self.hidden_states is None:
             self.init_hidden(bs)
        hidden_state = self.hidden_states.to(agent_inputs.device)

        # 3. 时间步循环 (Time Loop)
        # CommNet 必须逐帧运行，因为 t 时刻的通信依赖于 t 时刻的中间状态
        agent_outs_list = []
        
        for i in range(max_t):
            # 取出当前时刻的输入 [bs, n_agents, dim]
            inputs_t = agent_inputs[:, i] 
            
            # 【关键修改点 1】：必须展平 (Batch Folding)
            # 变成 [bs * n_agents, dim]，这样 GRUCell 才能吃进去
            inputs_t = inputs_t.reshape(-1, inputs_t.shape[-1])
            
            # hidden_state 也需要展平 (虽然你的 Agent 内部好像有 reshape，但在外面统一处理更安全)
            # 假设 hidden_state 是 [bs, n_agents, hidden_dim]
            # h_in 变成 [bs * n_agents, hidden_dim]
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)

            # --- 阶段 A: 编码 Observation (Encoder) ---
            # 现在的 inputs_t 和 h_in 都是 2D 的了，Agent 不会报错了
            h = self.agent.encode(inputs_t, h_in)
            
            # 【关键修改点 2】：通信前必须还原维度
            # 因为算均值需要区分哪些是同一个 Episode 的队友
            # 变回 [bs, n_agents, hidden_dim]
            h = h.view(bs, self.n_agents, -1)
            
            # --- 阶段 B: 通信循环 (K-Step Communication) ---
            K = getattr(self.args, "comm_steps", 1) 
            
            for k in range(K):
                # 1. 计算均值消息
                # Sum: [bs, 1, hidden_dim]
                h_sum = h.sum(dim=1, keepdim=True)
                
                # C_i = (Sum - h_i) / (N-1)
                if self.n_agents > 1:
                    c = (h_sum - h) / (self.n_agents - 1)
                else:
                    c = th.zeros_like(h)

                # 2. 融合消息 (Communicate Update)
                # 再次展平以通过全连接层: [bs * n_agents, dim]
                h_flat = h.reshape(-1, self.args.rnn_hidden_dim)
                c_flat = c.reshape(-1, self.args.rnn_hidden_dim)
                
                h_flat = self.agent.communicate(h_flat, c_flat)
                
                # 再次还原形状供下一轮计算: [bs, n_agents, dim]
                h = h_flat.view(bs, self.n_agents, -1)

            # --- 阶段 C: 解码动作 (Decoder) ---
            # 展平输入 Decoder
            out_t = self.agent.decode(h.reshape(-1, self.args.rnn_hidden_dim))
            
            # 保存这一步的输出
            agent_outs_list.append(out_t)
            
            # 更新 hidden state 给下一步使用 (保持 [bs, n_agents, hidden] 形状)
            hidden_state = h

        # 更新全局 hidden_states
        self.hidden_states = hidden_state
        
        # 拼接所有时间步的输出
        # List[[bs*n, n_actions]] -> [bs, max_t, n_agents, n_actions]
        agent_outs = th.stack(agent_outs_list, dim=1).view(bs, max_t, self.n_agents, -1)
        
        # 展平以适配后续处理 [bs * max_t * n_agents, n_actions]
        agent_outs = agent_outs.reshape(bs * max_t * self.n_agents, -1)

        # 4. Softmax 处理 (同 BasicMAC)
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * max_t * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, max_t, self.n_agents, -1)

    def init_hidden(self, batch_size):
        # 扩展维度 [bs, n_agents, hidden_dim]
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        # 【注意】这里需要在 agent_REGISTRY 里注册你的 CommNetAgent
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        
    def _build_inputs(self, batch, max_t, ts=slice(None)):
        # 保持与 BasicMAC 一致
        bs = batch.batch_size
        inputs = []
        if self.side == "red":
            inputs.append(batch["obs"][:, ts, :self.n_reds])
        else:
            inputs.append(batch["obs"][:, ts, self.n_reds:])

        if self.args.obs_last_action:
            if ts == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, ts]))
            else:
                inputs.append(batch["actions_onehot"][:, ts-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*max_t*self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        # 保持与 BasicMAC 一致
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape