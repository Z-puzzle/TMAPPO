from functools import partial
from .shipcombat_env import ShipCombatEnv,make_env,make_hunt_env


def env_fn(env, args_dict=None):  # 改为位置参数
    if args_dict is None:
        args_dict = {}
    return env(args_dict)

REGISTRY = {}
REGISTRY["shipcombat"] = partial(env_fn, env=make_env) 

REGISTRY["shiphunt"] = partial(env_fn, env=make_hunt_env)
