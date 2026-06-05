import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml
from run import run

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    for key,value in config.items():
        if isinstance(value, dict) and "env_args" in value:
            value["env_args"]["seed"] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)

def _is_centralized_learner(learner):
    return learner in {"mappo_learner", "comamappo_learner", "coma_learner"}

def _set_reward_mode(env_args, red_learner, blue_learner):
    n_reds = env_args.get("n_reds")
    n_blues = env_args.get("n_blues")
    if n_reds is None or n_blues is None:
        return
    env_args["n_reds_reward"] = 1 if _is_centralized_learner(red_learner) else n_reds
    env_args["n_blues_reward"] = 1 if _is_centralized_learner(blue_learner) else n_blues


if __name__ == '__main__':
    if len(sys.argv) <= 1:  # 无CLI参数,方便调试
        params = [
            "main.py",
            "--env-config=mv1combat",
            "--red-config=newmethod",
            "--blue-config=newmethod",
        ]
    else:
        params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "zpz.yaml"), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    red_alg_config = _get_config(params, "--red-config", "algs")
    blue_alg_config = _get_config(params, "--blue-config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    red_config_dict = deepcopy(config_dict)
    blue_config_dict = deepcopy(config_dict)
    red_config_dict = recursive_dict_update(red_config_dict, red_alg_config)
    blue_config_dict = recursive_dict_update(blue_config_dict, blue_alg_config)
    _set_reward_mode(red_config_dict["env_args"], red_config_dict["learner"], blue_config_dict["learner"])
    _set_reward_mode(blue_config_dict["env_args"], red_config_dict["learner"], blue_config_dict["learner"])

    # now add all the config to sacred
    ex.add_config({
        "red": red_config_dict,
        "blue": blue_config_dict,
    })

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)
