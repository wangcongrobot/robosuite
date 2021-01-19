

def test2():
    import numpy as np
    import robosuite as suite
    from robosuite.models import MujocoWorldBase
    world = MujocoWorldBase()

    from robosuite.models.robots import RexROV2UR3

    mujoco_robot = RexROV2UR3()

    from robosuite.models.grippers import gripper_factory

    gripper = gripper_factory("Robotiq140Gripper")
    # gripper.hide_visualization()
    mujoco_robot.add_gripper(gripper)

    mujoco_robot.set_base_xpos([-0.5, 0, 2.5])
    world.merge(mujoco_robot)

    from robosuite.models.arenas import UnderwaterArena
    mujoco_arena = UnderwaterArena()
    mujoco_arena.set_origin([0, 0, 0])
    world.merge(mujoco_arena)

    from robosuite.models.objects import DoorObject, BoxObject
    from robosuite.utils.mjcf_utils import new_joint

    # door = DoorObject(
    #     name="Door",
    #     friction=0.0,
    #     damping=0.1,
    #     lock=True,
    #     )
    # mujoco_object = door.get_obj()
    # mujoco_object.set('pos', '1.0 0 1.0')    
    # sphere.append(new_joint(name='sphere_free_joint', type='free'))
    # sphere.set('pos', '1.0 0 1.0')
    # world.worldbody.append(mujoco_object)

    model = world.get_model(mode="mujoco_py")
    # model.save_model()

    # add reference objects for x and y axes
    x_ref = BoxObject(name="x_ref", size=[0.01, 0.01, 0.01], rgba=[0, 1, 0, 1], obj_type="visual",
                        joints=None).get_obj()
    x_ref.set("pos", "0.2 0 0.105")
    world.worldbody.append(x_ref)
    y_ref = BoxObject(name="y_ref", size=[0.01, 0.01, 0.01], rgba=[0, 0, 1, 1], obj_type="visual",
                        joints=None).get_obj()
    y_ref.set("pos", "0 0.2 0.105")
    world.worldbody.append(y_ref)


    from mujoco_py import MjSim, MjViewer

    sim = MjSim(model)
    viewer = MjViewer(sim)
    viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

    for i in range(10000):
        action = np.random.randn(14)
        sim.data.ctrl[:] = action
        sim.step()
        viewer.render()


def test1():
    import numpy as np
    import robosuite as suite
    from robosuite.utils.mjcf_utils import save_sim_model    
    # create environment instance
    env = suite.make(
        env_name="UnderwaterValveVAE", # try with other tasks like "Stack" and "Door" UnderwaterValve
        robots="RexROV2UR3",  # try with other robots like "Sawyer" and "Jaco" RexROV2UR3
        gripper_types="Robotiq140Gripper",
        has_renderer=True,
        render_camera="frontview",
        has_offscreen_renderer=False,
        control_freq=20,
        horizon=2000,
        use_object_obs=False,
        use_camera_obs=False,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        reward_shaping=True,
    )

    # reset the environment
    env.reset()
    env.render()

    save_sim_model(env.sim, "model.xml")

    while True:
        print("dof: ", env.robots[0].dof)
        action = np.random.randn(env.robots[0].dof) # sample random action
        # for j in range(12):
            # action[j] = 0
        # action = np.zeros(13)
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display


def train():
    # general libraries
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import multiprocessing

    # robosuite libraries
    import robosuite as suite
    from robosuite.wrappers import GymWrapper


    # RL framework libraries
    from stable_baselines.common import make_vec_env
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.common.vec_env import DummyVecEnv
    from stable_baselines.common.vec_env import SubprocVecEnv
    from stable_baselines.common.vec_env import VecNormalize
    from stable_baselines.ppo2 import PPO2

    robot_name = "RexROV2UR3"
    env_name = "UnderwaterValve"
    gripper_type = "Robotiq140Gripper"
    log_dir = "log"
    folder_path = "/".join([log_dir, env_name + "-" + robot_name])

    # create environment instance
    env = suite.make(
        env_name = env_name,
        robots = robot_name,
        gripper_types = gripper_type,
        use_camera_obs = False,
        use_object_obs = True,
        reward_scale = None,
        reward_shaping = True,
        # use_indicator_object = False,
        has_renderer = False,
        has_offscreen_renderer = False,
        render_camera = "frontview",
        control_freq = 20,
        horizon = 500,
        ignore_done = False,
        hard_reset = False,
        camera_names = "agentview",
        camera_heights = 48,
        camera_widths = 48,
        camera_depths = False
    )

    gym_env = GymWrapper(env)
    n_envs = 25

    # make_vec_env produces `n_envs` parallel, monitored
    # environments for distributed training
    train_env = make_vec_env(
        env_id = lambda: gym_env,
        n_envs = n_envs,
        vec_env_cls = SubprocVecEnv, # default is DummyVecEnv in `make_vec_env`
        monitor_dir = folder_path + "/monitor"
    )

    # initialize PPO2 Model
    tensorboard_path = folder_path + "/tensorboard/"
    ppo2_model = PPO2(
        policy = MlpPolicy,
        env = train_env,
        verbose = 1,
        tensorboard_log=tensorboard_path,
        # max_grad_norm=0.5,
        # learning_rate=0.0003,
        # n_steps=1024,
        # nminibatches=16,
        # noptepochs=6,
        # vf_coef=0.5,
        # lam=0.95,
        # gamma=0.99,
        # cliprange=0.2,
        # ent_coef=0.0,
    )

    # train the model
    total_timesteps = int(2e6)
    ppo2_model.learn(
        total_timesteps=total_timesteps,
        log_interval=1,
        tb_log_name="SubprocVecEnv4Envs500Horizon"
    )

    # save model data
    model_filepath = folder_path + "/ppO2model4Envs500Horizon.pkl"
    ppo2_model.save(model_filepath)

    visualize_data(ppo2_model)


# looks at the individual rewards from each sub-environment
# after the PPO training
def visualize_data(ppo2_model):
    all_saved_rewards = ppo2_model.get_env().env_method("get_episode_rewards")
    env_number = 1
    for env_rewards in all_saved_rewards:
        episode_numbers = [i for i in range(len(env_rewards))]
        plt.plot(episode_numbers, env_rewards, label=f"Environment {env_number}", marker='o')
        env_number += 1

    plt.xlabel("Episode number")
    plt.ylabel("Reward")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    test1()
    # test2()    
    # train()
    print("Training completed.")