import torch as th
import torch.nn as nn
import torch.nn.functional as F

#这个是中心化评价网络，所以不分边。
class CentraQ(nn.Module):
    def __init__(self, scheme, args):
        super(CentraQ, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        #这里训练时，需要两边的q，所以这里还是要输出全部的q值。
        # observation，这个是面向两方的，是输出所有的q，则：
        inputs.append(batch["obs"][:, ts])
        # #只输出一方的q，则：
        # if self.side == "red":
        #     inputs.append(batch["red_obs"][:, ts])
        # else:      
        #     inputs.append(batch["blue_obs"][:, ts])


        # actions (masked out by agent)
        #每个智能体接受所有智能体的动作，
        # mlp对位置顺序是敏感的，是不是需要调整位置。
        actions = batch["actions_onehot"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)

        inputs.append(actions)

        # last actions，有没有上一步last actions
        # if t == 0:
        #     inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        # elif isinstance(t, int):
        #     inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        # else:
        #     last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
        #     last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        #     inputs.append(last_actions)

        #智能体编码
        # inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observation
        input_shape += scheme["obs"]["vshape"]
        # actions 
        input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        #and last actions
        # input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        # agent id 不是协同算法的不需要智能体编码
        # input_shape += self.n_agents
        return input_shape