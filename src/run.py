import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
import numpy as np



def run(_run, _config, _log):

    # check red_args sanity
    red_args = _config["red"]
    blue_args = _config["blue"]
    red_args = args_sanity_check(red_args, _log)
    blue_args = args_sanity_check(blue_args, _log)

    red_args = SN(**red_args)
    blue_args = SN(**blue_args)
    red_args.device = "cuda" if red_args.use_cuda else "cpu"
    blue_args.device = "cuda" if blue_args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    sacred_id = _run._id 

    # configure tensorboard logger

    unique_token = "{}_{}_{}_{}_{}".format(red_args.env,red_args.name,blue_args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),sacred_id)
    red_args.unique_token = unique_token
    if red_args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(all_args=[red_args,blue_args], logger=logger)
    # run_step(red_args=red_args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(red_args, runner):

    for _ in range(red_args.test_nepisode):
        runner.run(test_mode=True,options=True)

    if red_args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(all_args, logger):

    # Init runner so we can get env info
    
    red_args = all_args[0]
    blue_args = all_args[1]

    runner = r_REGISTRY[red_args.runner](args=all_args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    for arg in all_args:
        arg.n_reds = env_info["n_reds"]
        arg.n_blues = env_info["n_blues"]
        arg.n_actions = env_info["n_actions"]
        arg.n_reds_reward = arg.env_args["n_reds_reward"]
        arg.n_blues_reward = arg.env_args["n_blues_reward"]
    
    #这里还需要调整
    red_args.n_agents = red_args.n_reds
    blue_args.n_agents = blue_args.n_blues

    # Infer flattened per-ship observation dim from env wrappers (avoid hardcoding 11).
    obs_probe, _ = runner.env.reset()
    flatten_obs_probe = runner.env.get_flatten_obs(obs_probe)
    obs_vshape = int(flatten_obs_probe.shape[-1])



    # Default/Base scheme,用到的数据格式在这定义，设计出batch
    groups = {
        "ships":red_args.n_reds+red_args.n_blues,
        "reward":red_args.n_reds_reward+red_args.n_blues_reward,
        "state":1,
    }
    
    scheme = {
        "state": {"vshape": env_info["state_shape"], "group": "state"},
        "obs": {"vshape": obs_vshape, "group": "ships"},
        "actions": {"vshape": (1,), "group": "ships", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "ships", "dtype": th.int},
        #这里是分为两边，没边智能体的奖励相同，所以group为2.
        "reward": {"vshape": (1,), "group": "reward", "dtype": th.float32},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }

    #可以不要好像，不用onehot的话。
    #离散需要这个
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=red_args.n_actions)]),
    }

    buffer = ReplayBuffer(scheme, groups, red_args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if red_args.buffer_cpu_only else red_args.device)

    # Setup multiagent controller here
    if red_args.name != "rule":  
        red_mac = mac_REGISTRY[red_args.mac](buffer.scheme, groups, red_args)
    else:
        red_mac = mac_REGISTRY[red_args.mac](buffer.scheme, groups, red_args)
    if blue_args.name != "rule":            
        blue_mac = mac_REGISTRY[blue_args.mac](buffer.scheme, groups, blue_args)
    else:
        blue_mac = mac_REGISTRY[blue_args.mac](buffer.scheme, groups, blue_args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, red_mac=red_mac,blue_mac = blue_mac)

    # Learner
    learner1 = le_REGISTRY[red_args.learner](red_mac, buffer.scheme, logger, red_args)

    if red_args.use_cuda:
        learner1.cuda()

    #测试的时候，需要在main处把对应的算法args导入。否则会报错。
    #比如测试newmethod和ppo，必须在main para中导入newmethod 和ppo
    if red_args.checkpoint_path != "":
        #没有定义测试的模型，则设为规则智能体
        if red_args.checkpoint_path != "":

            timesteps = []
            timestep_to_load = 0

            if not os.path.isdir(red_args.checkpoint_path):
                logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(red_args.checkpoint_path))
                return

            # Go through all files in red_args.checkpoint_path
            for name in os.listdir(red_args.checkpoint_path):
                full_name = os.path.join(red_args.checkpoint_path, name)
                # Check if they are dirs the names of which are numbers
                if os.path.isdir(full_name) and name.isdigit():
                    timesteps.append(int(name))

            if red_args.load_step == 0:
                # choose the max timestep
                timestep_to_load = max(timesteps)
            else:
                # choose the timestep closest to load_step
                timestep_to_load = min(timesteps, key=lambda x: abs(x - red_args.load_step))

            red_model_path = os.path.join(red_args.checkpoint_path, str(timestep_to_load))

            logger.console_logger.info("Loading model from {}".format(red_model_path))
            if not red_args.test:
                learner1.load_models(red_model_path)
            else:
                learner1.load_mac(red_model_path)
            #这个会影响test吗？
            runner.t_env = timestep_to_load

        if red_args.evaluate or red_args.save_replay:
            evaluate_sequential(red_args, runner)
            episode = 0
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()

            return

    # start training
    episode = 0
    last_test_T = 0
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(red_args.t_max))

    #定义成了回调函数，传递这个函数给
    def train_fn(episode_sample, t_env, episode):
        red_batch = split_batch(episode_sample, red_args.n_reds, red_args.n_reds_reward)
        learner1.train(red_batch, t_env, episode)
        
    while runner.t_env <= red_args.t_max:

        # Run for a whole episode at a time

        episode_batch = runner.run(test_mode=False,train_callback=train_fn)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(red_args.batch_size):
            episode_sample = buffer.sample(red_args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != red_args.device:
                episode_sample.to(red_args.device)

            train_fn(episode_sample, runner.t_env, episode)
            if red_args.policy_type =="on_policy":
                buffer.clear()

        # Execute test runs once in a while
        #test_interval间隔进行测试，测试跑test_nepisode次
        #这个可以是这个时间点的模型性能
        
        if (runner.t_env - last_test_T) >= red_args.test_interval or last_test_T==0:
            n_test_runs = max(1, red_args.test_nepisode // runner.batch_size)

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, red_args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, red_args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=False,options=True)

        if red_args.save_model and (runner.t_env - model_save_time >= red_args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(red_args.local_results_path, "models", red_args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner1.save_models(save_path)

        episode += red_args.batch_size_run

        #log_interval间隔，打印一次训练结果
        if (runner.t_env - last_log_T) >= red_args.log_interval or last_log_T==0:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")

def split_batch(batch, n_reds, n_reds_reward):
    # --- 1. 创建红方 Batch ---
    red_batch = batch.copy()
    
    # 修改 Obs: 只保留前 n_reds 个智能体
    # [bs, t, n_all, dim] -> [bs, t, n_reds, dim]
    red_batch.data.transition_data["obs"] = batch["obs"][:, :, :n_reds]
    
    # 修改 Actions
    red_batch.data.transition_data["actions"] = batch["actions"][:, :, :n_reds]

    red_batch.data.transition_data["actions_onehot"] = batch["actions_onehot"][:, :, :n_reds]
    
    # 修改 Avail Actions (如果有)
    if "avail_actions" in batch.data.transition_data:
        red_batch.data.transition_data["avail_actions"] = batch["avail_actions"][:, :, :n_reds]

    # 修改 Reward: 取第0个维度的 side (假设 0 是红方)
    # [bs, t, 2, 1] -> [bs, t, 1, 1] 保持维度以便 learner 处理
    red_batch.data.transition_data["reward"] = batch["reward"][:, :, :n_reds_reward]

    # 注意：State (全局状态) 通常不需要切分，红蓝双方共享同一个 Global State
    # 如果需要切分 State，也可以在这里处理
    
    return red_batch


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
