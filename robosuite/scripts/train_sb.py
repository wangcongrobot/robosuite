import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, SAC, DDPG

import numpy as np
import robosuite as suite
from robosuite.utils.mjcf_utils import save_sim_model    
from robosuite.wrappers.gym_wrapper import GymWrapper
# create environment instance
env = suite.make(
    env_name="UnderwaterValve", # try with other tasks like "Stack" and "Door" UnderwaterValve
    robots="RexROV2UR3",  # try with other robots like "Sawyer" and "Jaco" RexROV2UR3
    gripper_types="Robotiq140Gripper",
    has_renderer=False,
    render_camera="frontview",
    has_offscreen_renderer=True,
    control_freq=20,
    horizon=200,
    use_object_obs=True,
    use_camera_obs=False,
    camera_names="agentview",
    camera_heights=84,
    camera_widths=84,
    reward_shaping=True,
)

env = GymWrapper(env)

# reset the environment
env.reset()
# env.render()

save_sim_model(env.sim, "model.xml")

# for i in range(10000):
#     print("dof: ", env.robots[0].dof)
#     action = np.random.randn(env.robots[0].dof) # sample random action
#     # for j in range(12):
#         # action[j] = 0
#     # action = np.zeros(13)
#     obs, reward, done, info = env.step(action)  # take action in the environment
#     env.render()  # render on display


# env = gym.make('CartPole-v1')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])



# env = suite.make(
#     env_name="UnderwaterValve", # try with other tasks like "Stack" and "Door" UnderwaterValve
#     robots="RexROV2UR3",  # try with other robots like "Sawyer" and "Jaco" RexROV2UR3
#     gripper_types="Robotiq140Gripper",
#     has_renderer=True,
#     render_camera="frontview",
#     has_offscreen_renderer=False,
#     control_freq=20,
#     horizon=200,
#     use_object_obs=True,
#     use_camera_obs=False,
#     camera_names="agentview",
#     camera_heights=84,
#     camera_widths=84,
#     reward_shaping=True,
# )

# env = GymWrapper(env)

# model = DDPG(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=int(1e5))
# model.save("rexrov_ddpg")

model_sac = SAC(MlpPolicy, env, verbose=1)
model_sac.learn(total_timesteps=int(1e5))
model_sac.save("rexrov_sac")

# model.load("rexrov_ppo2")


# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

env.close()


# import gym
# import numpy as np

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import SubprocVecEnv
# from stable_baselines.common import set_global_seeds, make_vec_env
# from stable_baselines import ACKTR

# def make_env(env_id, rank, seed=0):
#     """
#     Utility function for multiprocessed env.

#     :param env_id: (str) the environment ID
#     :param num_env: (int) the number of environments you wish to have in subprocesses
#     :param seed: (int) the inital seed for RNG
#     :param rank: (int) index of the subprocess
#     """
#     def _init():
#         env = gym.make(env_id)
#         env.seed(seed + rank)
#         return env
#     set_global_seeds(seed)
#     return _init

# if __name__ == '__main__':
#     env_id = "CartPole-v1"
#     num_cpu = 4  # Number of processes to use
#     # Create the vectorized environment
#     env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

#     # Stable Baselines provides you with make_vec_env() helper
#     # which does exactly the previous steps for you:
#     # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

#     model = ACKTR(MlpPolicy, env, verbose=1)
#     model.learn(total_timesteps=25000)

#     obs = env.reset()
#     for _ in range(1000):
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = env.step(action)
#         env.render()
