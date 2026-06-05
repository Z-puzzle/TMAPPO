import torch as th
import torch.nn as nn
import torch.nn.functional as F

#这个是中心化评价网络，所以不分边。
class CentraV(nn.Module):
    def __init__(self, scheme, args, n_agents=1):
        super(CentraV, self).__init__()

        self.args = args
        self.n_agents = n_agents
        self.n_ships = args.n_reds + args.n_blues
        self.n_actions = args.n_actions

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

        # state: (bs, T, state_dim) -> (bs, T, 1, state_dim)
        # inputs.append(batch["state"][:, ts].repeat(1, 1, self.n_agents,1))
        inputs.append(batch["state"][:, ts])

        # obs: (bs, T, n_ships, obs_dim) -> (bs, T, 1, n_ships*obs_dim)
        # inputs.append(batch["obs"][:, ts])

        # inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        inputs = th.cat([x.reshape(bs, max_t, 1, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # joint observation (all ships flattened)
        # input_shape += scheme["obs"]["vshape"]

        # agent id
        # input_shape += self.n_agents
        return input_shape
