REGISTRY = {}

from .rnn_agent import RNNAgent
from .agent import Agent
from .normal_agent import NormalAgent
from .commnet_agent import CommNetAgent
from .transformer_agent import TransformerAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["nn"] =Agent
REGISTRY["normalnn"] = NormalAgent
REGISTRY["commagent"] = CommNetAgent
REGISTRY["transformeragent"] = TransformerAgent