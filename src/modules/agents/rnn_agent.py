import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, scheme, args):
        super(RNNAgent, self).__init__()
        self.args = args

        input_shape = self._get_input_shape(scheme)

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRU(args.rnn_hidden_dim, args.rnn_hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, ep_batch, hidden_state,t=None):
            inputs = self._build_inputs(ep_batch,t)

            b, t, a, f = inputs.size()

            # 1. 只有在 T > 1 时 transpose 才是物理必须的
            # 但为了代码统一，这样写最稳健。PyTorch 会在必要时自动处理内存连续性
            x = inputs.transpose(1, 2).reshape(b * a, t, f)
            
            x = F.relu(self.fc1(x))
            
            # 2. 这里的 .contiguous() 是为了解决你之前的 GRU 报错
            h_in = hidden_state.reshape(1, b * a, -1).contiguous()
            
            # 3. GRU 内部计算才是真正的“大头”开销
            rnn_out, h_n = self.rnn(x, h_in)
            
            # 4. Q 值计算
            q = self.fc2(rnn_out)
            
            # 5. 还原维度
            q = q.view(b, a, t, -1).transpose(1, 2)
            
            return q, h_n

    def _build_inputs(self, batch,ts=None):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        ts = slice(None) if ts is None else slice(ts, ts + 1)
        inputs = []
        #这里都是考虑共享参数，即一边只有一个mac，所有的obs通过一个参数mac。如果参数不共享，比如各自agent的ippo，这个就不适用。
        inputs.append(batch["obs"][:, ts])  # b1av


        if self.args.obs_last_action:
            if ts == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, ts]))
            else:
                inputs.append(batch["actions_onehot"][:, ts-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        #要输入到rnn，必须是2d，bs*智能体数量，-1，输入的是某一方所有智能体的观测。输出所有智能体的动作。

        inputs = th.cat([x for x in inputs], dim=-1)

        return inputs
    
    #如果transformer的输入不和这个一样，就不用这个，内部再自己定义好了。不用这个inputshape不就完了构建就是了。
    def _get_input_shape(self, scheme):
        #两边船的obsshape应该是一致的
        #if red:
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
