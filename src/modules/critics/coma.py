import torch as th
import torch.nn as nn
import torch.nn.functional as F

#这个是中心化评价网络，所以不分边。
class COMACritic(nn.Module):
    def __init__(self, scheme, args,side):
        super(COMACritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.side = side
        if self.side =="red":
            self.n_agents = args.n_reds
        else:
            self.n_agents = args.n_blues

        s_input_shape = self._get_s_input_shape(scheme)
        u_input_shape = self._get_u_input_shape(scheme)

        self.output_type = "q"

        # Set up network layers
        self.s_fc1 = nn.Linear(s_input_shape, 128)
        self.u_fc1 = nn.Linear(u_input_shape, 64)
        self.fc2 = nn.Linear(128+64, 128)
        self.fc3 = nn.Linear(128, self.n_actions)

    def forward(self, batch, t=None):
        s_inputs = self._build_s_inputs(batch, t=t)
        u_inputs = self._build_u_inputs(batch, t=t)
        sf = F.relu(self.s_fc1(s_inputs))
        uf = F.relu(self.u_fc1(u_inputs))
        x = th.cat([sf, uf], dim=-1)
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_s_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        #这里训练时，需要两边的q，所以这里还是要输出全部的q值。
        # observation，这个是面向两方的，是输出所有的q，则：
        #compete这个参数是针对learner的。
        # 还是环境？有可能一边是竞争，一边是协同吗？可能是很后面的需求了。一边训练协同包夹，另一边破包夹。两船，我的方法可行。另一边是多船，固定多个对手动作，输出
        if self.args.compete:
            inputs.append(batch["obs"][:, ts])
        # # #不考虑蓝方是智能体
        # else:
        #     if self.side == "red":
        #         inputs.append(batch["obs"][:, ts,:self.n_agents])
        #     else:      
        #         inputs.append(batch["obs"][:, ts,-self.n_agents:])


        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _build_u_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []

        # actions (masked out by agent)
        #每个智能体接受所有智能体的动作，屏蔽了自己的动作，mlp对位置顺序是敏感的，所以有效。
        #屏蔽了自己动作的所有动作输入
        actions = batch["actions_onehot"][:, ts,:self.n_agents].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))

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


    def _get_s_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observation
        input_shape += scheme["obs"]["vshape"]

        # agent id 不是协同算法的不需要智能体编码
        # input_shape += self.n_agents
        return input_shape
    
    def _get_u_input_shape(self, scheme):
        # actions 
        input_shape = scheme["actions_onehot"]["vshape"][0] * self.n_agents
        #and last actions
        # input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        return input_shape