from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from torch import nn


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.args = args
        self.n_reds = args.n_reds
        self.n_blues = args.n_blues
        self.n_agents = args.n_agents     

        
        self._build_agents(scheme)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        #传递一个self.agent
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t=None, test_mode=False):

        max_t = ep_batch.max_seq_length if t is None else 1
        
        avail_actions = ep_batch["avail_actions"][:, t]
        if self.args.agent:
            agent_outs, self.hidden_states = self.agent(ep_batch, self.hidden_states,t)
        else: #zpz
            agent_outs = self.agent(ep_batch)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size,max_t, self.n_agents, -1)


    def init_hidden(self, batch_size):
        if self.args.agent:
            #expand是扩展视图！不是真正的复制，所以这里要用repeat！
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).repeat(batch_size, self.n_agents, 1)  # bav
        else:
            pass


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

    def _build_agents(self, scheme):

        self.agent = agent_REGISTRY[self.args.agent](scheme, self.args)
        
