REGISTRY = {}

from .basic_controller import BasicMAC
from .rule_controller import RuleMac
from .corp_controller import CorpMac
from .commnet_controller import CommNetMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["rule_mac"] = RuleMac
REGISTRY["corp_mac"] = CorpMac
REGISTRY["comm_mac"] = CommNetMAC