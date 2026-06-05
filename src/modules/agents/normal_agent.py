import torch.nn as nn
import torch.nn.functional as F
import torch


class NormalAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NormalAgent, self).__init__()
        self.args = args

        self.log_std_init = -0.5

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.log_std = nn.Parameter(torch.ones(args.n_actions) * self.log_std_init, requires_grad=True)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        mu = self.fc2(h)
        return mu, h
