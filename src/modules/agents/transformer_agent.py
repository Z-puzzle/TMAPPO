import torch as th
import torch.nn as nn
import torch.nn.functional as F

class InterpretableTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x, mask=None):
        res = x
        x = self.norm1(x)
        # attn_w 可用于后续的虚实融合态势分析
        attn_out, attn_w = self.self_attn(x, x, x, key_padding_mask=mask, need_weights=True)
        x = res + attn_out
        res = x
        x = self.norm2(x)
        x = res + self.ffn(x)
        return x, attn_w

class TransformerAgent(nn.Module):
    def __init__(self, scheme, args):
        super(TransformerAgent, self).__init__()
        self.args = args
        self.n_ships = self.args.n_reds + self.args.n_blues
        self.n_agents = self.args.n_agents 
        if self.n_agents == self.args.n_reds:
            agent_start = 0
        elif self.n_agents == self.args.n_blues:
            agent_start = self.args.n_reds
        else:
            agent_start = 0
        self.register_buffer("self_ship_idx", th.arange(self.n_agents) + agent_start, persistent=False)

        obs_vshape = scheme.get("obs", {}).get("vshape")
        if isinstance(obs_vshape, (tuple, list)):
            obs_vshape = int(obs_vshape[0])
        obs_vshape = int(obs_vshape)
        if obs_vshape % self.n_ships != 0:
            raise ValueError(f"obs_vshape={obs_vshape} not divisible by n_ships={self.n_ships}")
        self.feat_dim = obs_vshape // self.n_ships
        input_shape = self.feat_dim
        self.model_dim = args.rnn_hidden_dim

        state_vshape = scheme.get("state", {}).get("vshape")
        if isinstance(state_vshape, (tuple, list)):
            state_vshape = int(state_vshape[0])
        state_vshape = int(state_vshape)
        if state_vshape % self.n_ships != 0:
            raise ValueError(f"state_shape={state_vshape} not divisible by n_ships={self.n_ships}")
        self.state_feat_dim = state_vshape // self.n_ships

        self.input_proj = nn.Linear(input_shape, self.model_dim)
        
        self.transformer_layers = nn.ModuleList([
            InterpretableTransformerLayer(self.model_dim, args.n_heads, self.model_dim * 4)
            for _ in range(args.n_layers)
        ])
        
        # 输入维度：Transformer 特征 + state 自身特征
        self.rnn = nn.GRU(self.model_dim + self.state_feat_dim, args.rnn_hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # 【修改点 1】修复 fc1 不存在的错误，改用 input_proj 或 fc2
        # 使用 weight.new_zeros 确保设备和数据类型自动继承
        return self.input_proj.weight.new_zeros(1, self.args.rnn_hidden_dim)

    def forward(self, batch, hidden_state, t=None):
        inputs = self._build_inputs(batch, t=t) # [B, T, n_agents, n_ships, feat_dim]
        b, t_len, n_agents, n_ships, feat_dim = inputs.shape
        
        # --- 1. Transformer 空间博弈 ---
        inputs_reshaped = inputs.reshape(b * t_len * n_agents, n_ships, feat_dim)
        x = F.relu(self.input_proj(inputs_reshaped)) 
        
        for layer in self.transformer_layers:
            x, _ = layer(x) # x: [B*T*n_agents, n_ships, D]
            
        # --- 2. 提取当前智能体并进行特征注入 ---
        x = x.view(b, t_len, n_agents, n_ships, -1)
        agent_idx = th.arange(n_agents, device=inputs.device)
        ship_idx = self.self_ship_idx.to(inputs.device)
        x_self = x[:, :, agent_idx, ship_idx, :]

        ts = slice(None) if t is None else slice(t, t+1)
        own_self = batch["state"][:, ts]
        own_self = own_self.view(b, t_len, n_ships, -1)
        own_self = own_self[:, :, agent_idx, :]

        x_combined = th.cat([x_self, own_self], dim=-1) # [B, T, n_agents, D+S]

        # --- 3. 重组维度准备进入 GRU ---
        x_combined = x_combined.view(b, t_len, self.n_agents, -1)
        # 交换 T 和 A 维度，确保 GRU 处理的是同一个 Agent 的时间序列
        x_gru_in = x_combined.transpose(1, 2).reshape(b * self.n_agents, t_len, -1)
        
        # --- 4. GRU 处理 ---
        # 【修改点 2】强制设备对齐，防止 Device Mismatch
        hidden_state = hidden_state.reshape(1, b * self.n_agents, -1).to(inputs.device)
        
        hidden_state = hidden_state.contiguous()
        output, h_n = self.rnn(x_gru_in, hidden_state)
        
        # --- 5. 输出 Q 值 ---
        q = self.fc2(output) 
        # 还原回标准格式: [B, T, n_agents, n_actions]
        q = q.view(b, self.n_agents, t_len, -1).transpose(1, 2)

        return q, h_n
        
    def _build_inputs(self, batch, t=None):
        ts = slice(None) if t is None else slice(t, t+1)

        obs = batch["obs"][:, ts]
        bs, t_len = obs.shape[0], obs.shape[1]
        if obs.shape[2] == self.n_agents:
            obs_agents = obs
        elif obs.shape[2] == self.n_ships:
            obs_agents = obs[:, :, self.self_ship_idx]
        else:
            raise ValueError(f'Unexpected obs.shape[2]={obs.shape[2]} (expected {self.n_agents} or {self.n_ships})')
        return obs_agents.reshape(bs, t_len, self.n_agents, self.n_ships, self.feat_dim)
