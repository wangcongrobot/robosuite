import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import numpy as np
import robosuite as suite
from robosuite.utils.mjcf_utils import save_sim_model    
from robosuite.wrappers.gym_wrapper import GymWrapper
# create environment instance
env = suite.make(
    env_name="Door", # try with other tasks like "Stack" and "Door" UnderwaterValve
    robots="UR5e",  # try with other robots like "Sawyer" and "Jaco" RexROV2UR3
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
    # camera_depth=False,
    reward_shaping=True,
)

env = GymWrapper(env)

# reset the environment
# env.reset()
# env.render()

# save_sim_model(env.sim, "model.xml")

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

model = PPO2(MlpPolicy, env, verbose=1)
# # train the agent
model.learn(total_timesteps=int(1e6))
# # save the model
model.save("ur5e_door_ppo2")
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

import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines import DDPG
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

# Create log dir
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = GymWrapper(env)
env = Monitor(env, log_dir)

# Add some param noise for exploration
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
# Because we use parameter noise, we should use a MlpPolicy with layer normalization
model = DDPG(LnMlpPolicy, env, param_noise=param_noise, verbose=0)
# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
# Train the agent
time_steps = 1e5
model.learn(total_timesteps=int(time_steps), callback=callback)

results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "UR5e_Door")
plt.show()