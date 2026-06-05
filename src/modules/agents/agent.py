import torch.nn as nn
import torch.nn.functional as F
import torch as th


class Agent(nn.Module):
    def __init__(self, input_shape, args):
        super(Agent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)

        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))

        q = self.fc2(x)
        return q
