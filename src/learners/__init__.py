from .coma_learner import COMALearner
from .rule_learner import RuleLearner
from .RCCMAlearner import RCCMAlearner
from .mappo_learner import MAPPOLearner
from .q_learner import QLearner
from .IACQ_learner import IACQ_learner
from .coma_mappo import COMAMAPPOLearner
from .Ippo_learner import IPPOLearner

REGISTRY = {}

REGISTRY["coma_learner"] = COMALearner
REGISTRY["rule_learner"] = RuleLearner
REGISTRY["mappo_learner"] = MAPPOLearner
REGISTRY["ippo_learner"] = IPPOLearner
REGISTRY["q_learner"] = QLearner
REGISTRY["RCCMA_learner"] = RCCMAlearner
REGISTRY["IACQ_learner"] = IACQ_learner
REGISTRY["comamappo_learner"] = COMAMAPPOLearner
