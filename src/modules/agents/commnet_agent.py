import torch
import torch.nn as nn
import torch.nn.functional as F

class CommNetAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CommNetAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents  # 必须知道有多少个智能体才能还原维度

        # 1. 基础网络结构
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        
        # 2. 通信层
        # 输入: [h_self, c_mean] -> 输出: h_new
        self.comm_update = nn.Linear(args.rnn_hidden_dim * 2, args.rnn_hidden_dim)
        
        # 3. 输出层
        self.action_head = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # 配合 BasicMAC，返回 (1, hidden_dim)
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        """
        这个 forward 会被 BasicMAC 在每个时间步调用一次。
        inputs: [batch_size * n_agents, input_dim]
        hidden_state: [batch_size * n_agents, hidden_dim]
        """
        
        # --- 1. 常规 RNN 处理 (Encoder) ---
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in) # 此时 h 的形状: [bs * n, hidden_dim]

        # --- 2. 内部通信黑魔法 (Internal Communication) ---
        # 这里的关键是：虽然输入被压扁了，但我们在 Agent 内部把它还原回去进行通信
        
        # 推断 batch_size
        # inputs.shape[0] 是 bs * n_agents
        bs = inputs.shape[0] // self.n_agents
        
        # 还原维度: [bs * n, dim] -> [bs, n, dim]
        h_reshaped = h.view(bs, self.n_agents, -1)
        
        # 读取通信步数 K
        K = getattr(self.args, "comm_steps", 1) 
        
        for k in range(K):
            # A. 计算所有智能体的均值消息
            # sum: [bs, 1, dim]
            h_sum = h_reshaped.sum(dim=1, keepdim=True)
            
            # C_i = (Sum - h_i) / (N-1)
            # 排除自己，计算其他人的平均值
            if self.n_agents > 1:
                c = (h_sum - h_reshaped) / (self.n_agents - 1)
            else:
                c = torch.zeros_like(h_reshaped)
            
            # B. 准备通信更新
            # 我们需要把 tensors 再次压扁去过 Linear 层 (PyTorch Linear 喜欢 2D 输入)
            # [bs, n, dim] -> [bs * n, dim]
            h_flat = h_reshaped.reshape(-1, self.args.rnn_hidden_dim)
            c_flat = c.reshape(-1, self.args.rnn_hidden_dim)
            
            # C. 融合信息
            combined = torch.cat([h_flat, c_flat], dim=-1)
            h_updated = torch.tanh(self.comm_update(combined))
            
            # D. 更新状态供下一轮通信使用
            h_reshaped = h_updated.view(bs, self.n_agents, -1)
        
        # 通信结束后的最终状态
        h_final = h_reshaped.reshape(-1, self.args.rnn_hidden_dim)

        # --- 3. 输出动作 (Decoder) ---
        q_values = self.action_head(h_final)

        # 返回: 动作值, 新的隐状态
        # BasicMAC 会自动把这个 h_final 存起来传给下一时刻
        return q_values, h_final