#!/usr/bin/env python
# encoding: utf-8

from robosuite.wrappers.gym_wrapper import GymWrapper
from ray.tune.registry import register_env
import gym
import os
import ray
import ray.rllib.agents.ppo as ppo
import shutil

import robosuite as suite

import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os

import ray
from ray import tune
from ray.tune import grid_search
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved

def main ():
    # register the custom environment
    select_env = "rexrov2_ur3_valve"
    
    # create environment instance
    env = suite.make(
        env_name="UnderwaterValve", # try with other tasks like "Stack" and "Door" UnderwaterValve
        robots="RexROV2UR3",  # try with other robots like "Sawyer" and "Jaco" RexROV2UR3
        gripper_types="Robotiq140Gripper",
        has_renderer=False,
        render_camera="frontview",
        has_offscreen_renderer=False,
        control_freq=20,
        horizon=200,
        use_object_obs=True,
        use_camera_obs=False,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        reward_shaping=True,
    )

    register_env(select_env, lambda config: env)

    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    agent = ppo.PPOTrainer(config, env=select_env)

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    n_iter = 5

    # train a policy with RLlib using PPO
    for n in range(n_iter):
        result = agent.train()
        chkpt_file = agent.save(chkpt_root)

        # print(status.format(
        #         n + 1,
        #         result["episode_reward_min"],
        #         result["episode_reward_mean"],
        #         result["episode_reward_max"],
        #         result["episode_len_mean"],
        #         chkpt_file
        #         ))


    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())


    # apply the trained policy in a rollout
    agent.restore(chkpt_file)
    gym_env = GymWrapper(env)
    # env = gym.make(select_env)

    state = env.reset()
    sum_reward = 0
    n_step = 20

    # for step in range(n_step):
    #     action = agent.compute_action(state)
    #     state, reward, done, info = env.step(action)
    #     sum_reward += reward

    #     env.render()

    #     if done == 1:
    #         # report at the end of each episode
    #         print("cumulative reward", sum_reward)
    #         state = env.reset()
    #         sum_reward = 0


    # init directory in which to save checkpoints
    chkpt_root = "log/rllib/"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)


    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True)
    args = parser.parse_args()
    ray.init()

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel if args.torch else CustomModel)

    config = {
        "env": "rexrov2_ur3_valve",  # or "corridor" if registered above
        # "env_config": {
        #     "corridor_length": 5,
        # },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        # "model": {
        #     "custom_model": "my_model",
        # },
        "vf_share_layers": True,
        "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
        "num_workers": 25,  # parallelism
        "framework": "torch" if args.torch else "tf",
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        # "episode_reward_mean": args.stop_reward,
    }

    results = tune.run(args.run, config=config, stop=stop)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()