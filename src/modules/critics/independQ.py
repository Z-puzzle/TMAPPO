import torch as th
import torch.nn as nn
import torch.nn.functional as F

#这个不是中心化的评价网络，所以需要分边。Q网络输出动作维度
class IndependQ(nn.Module):
    def __init__(self, scheme, args,side):
        super(IndependQ, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_reds = args.n_reds
        self.n_blues = args.n_blues
        self.side = side

        if self.side =="red":
            self.n_agents = args.n_reds
        else:
            self.n_agents = args.n_blues

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"


        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.n_actions)

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch, t=None):
        #这是单智能体，v网络只需输入自己的观测。
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # #只输出一方的q，则：
        if self.side == "red":
            inputs.append(batch["obs"][:, ts,:self.n_reds])
        else:      
            inputs.append(batch["obs"][:, ts,self.n_reds:])

        #智能体编码
        # inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):

        # observation
        input_shape = scheme["obs"]["vshape"]
        # # agent id
        # input_shape += self.n_agents

        return input_shape