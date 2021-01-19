from collections import OrderedDict
import numpy as np

# env task
from robosuite.models.arenas import TableArena, UnderwaterArena
from robosuite.models.objects import ValveObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor

# single_arm_env
from robosuite.robots import SingleArm
from robosuite.utils.transform_utils import mat2quat

# manipulation_env
from robosuite.environments.robot_env import RobotEnv
from robosuite.models.grippers import GripperModel
from robosuite.models.base import MujocoModel
from robosuite.robots import Manipulator, ROBOT_CLASS_MAPPING

class UnderwaterValveVAE(RobotEnv):
    """
    This class corresponds to the valve turning task for a underwater single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        use_latch (bool): if True, uses a spring-loaded handle and latch to "lock" the door closed initially
            Otherwise, door is instantiated with a fixed handle

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        # use_latch=True,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        use_vae=False,
        vae=None,
    ):

        # Robot info
        robots = list(robots) if type(robots) is list or type(robots) is tuple else [robots]
        num_robots = len(robots)

        # Gripper
        gripper_types = self._input2list(gripper_types, num_robots)

        # Robot configurations to pass to super call
        robot_configs = [
            {
                "gripper_type": gripper_types[idx],
            }
            for idx in range(num_robots)
        ]

        # settings for table top (hardcoded since it's not an essential part of the environment)
        self.table_full_size = (0.8, 0.3, 0.05)
        self.table_offset = (-0.2, -0.35, 0.8)

        self.robot_base_pos = [0, 1.5, 1.5]
        self.robot_base_ori = [0, 0, -1.57]

        # reward configuration
        # self.use_latch = use_latch
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        if (use_vae==True and vae is not None):
            self.vae = vae
            self.z_size = vae.z_size


        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            # gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            # use_vae=False,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided if the valve is turned

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 0.25], proportional to the distance between valve handle and robot arm
            - Rotating: in [0, 0.25], proportional to angle rotated by valve handled

        Note that a successfully completed task (valve turned) will return 1.0 irregardless of whether the environment
        is using sparse or shaped rewards

        Note that the final reward is normalized and scaled by reward_scale / 1.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # else, we consider only the case if we're using shaped rewards
        elif self.reward_shaping:
            # Add reaching component
            dist = np.linalg.norm(self._gripper_to_handle)
            print("_gripper_to_handle dist: ", dist)
            reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist)) # [0, 0.25]
            reward += reaching_reward
            # Add rotating component if we're using a locked door
            # if self.use_latch:
            handle_qpos = self.sim.data.qpos[self.handle_qpos_addr]
            reward += np.clip(0.25 * np.abs(handle_qpos / (0.5 * np.pi)), -0.25, 0.25)
            print("reaching_reward: ", reaching_reward)
            print("rotating reward: ", np.clip(0.25 * np.abs(handle_qpos / (0.5 * np.pi)), -0.25, 0.25))

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0
        print("total reward: ", reward)
        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Verify the correct robot has been loaded
        # assert isinstance(self.robots[0], SingleArm), \
        #     "Error: Expected one single-armed robot! Got {} type instead.".format(type(self.robots[0]))

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(self.robot_base_pos)
        self.robots[0].robot_model.set_base_ori(self.robot_base_ori)

        # load model for table top workspace
        # mujoco_arena = TableArena(
        #     table_full_size=self.table_full_size,
        #     table_offset=self.table_offset,
        # )
        mujoco_arena = UnderwaterArena()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
        )

        # initialize objects of interest
        self.valve = ValveObject(
            name="Valve",
            friction=0.0,
            damping=0.1,
            # lock=self.use_latch,
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.valve)
        else:
            self.placement_initializer = UniformRandomSampler(
                    name="ObjectSampler",
                    mujoco_objects=self.valve,
                    x_range=[0.07, 0.09],
                    y_range=[-0.01, 0.01],
                    rotation=(-np.pi / 2. - 0.25, -np.pi / 2.),
                    rotation_axis='z',
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=self.valve,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.object_body_ids = dict()
        self.object_body_ids["valve"] = self.sim.model.body_name2id(self.valve.valve_body)
        self.object_body_ids["frame"] = self.sim.model.body_name2id(self.valve.frame_body)
        self.object_body_ids["latch"] = self.sim.model.body_name2id(self.valve.latch_body)
        self.valve_handle_site_id = self.sim.model.site_name2id(self.valve.important_sites["handle"])
        # self.hinge_qpos_addr = self.sim.model.get_joint_qpos_addr(self.valve.joints[0])
        # if self.use_latch:
        self.handle_qpos_addr = self.sim.model.get_joint_qpos_addr(self.valve.joints[0])

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # Define sensor callbacks
            @sensor(modality=modality)
            def valve_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.object_body_ids["valve"]])

            @sensor(modality=modality)
            def handle_pos(obs_cache):
                return self._handle_xpos

            @sensor(modality=modality)
            def valve_to_eef_pos(obs_cache):
                return obs_cache["valve_pos"] - obs_cache[f"{pf}eef_pos"] if\
                    "valve_pos" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)

            @sensor(modality=modality)
            def handle_to_eef_pos(obs_cache):
                return obs_cache["handle_pos"] - obs_cache[f"{pf}eef_pos"] if\
                    "handle_pos" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)

            # @sensor(modality=modality)
            # def hinge_qpos(obs_cache):
            #     return np.array([self.sim.data.qpos[self.hinge_qpos_addr]])

            # sensors = [valve_pos, handle_pos, valve_to_eef_pos, handle_to_eef_pos, hinge_qpos]
            sensors = [valve_pos, handle_pos, valve_to_eef_pos, handle_to_eef_pos]
            names = [s.__name__ for s in sensors]

            # Also append handle qpos if we're using a locked door version with rotatable handle
            # if self.use_latch:
            @sensor(modality=modality)
            def handle_qpos(obs_cache):
                return np.array([self.sim.data.qpos[self.handle_qpos_addr]])
            sensors.append(handle_qpos)
            names.append("handle_qpos")

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # We know we're only setting a single object (the door), so specifically set its pose
            valve_pos, valve_quat, _ = object_placements[self.valve.name]
            valve_body_id = self.sim.model.body_name2id(self.valve.root_body)
            self.sim.model.body_pos[valve_body_id] = valve_pos
            self.sim.model.body_quat[valve_body_id] = valve_quat

    def _check_success(self):
        """
        Check if valve has been turned.

        Returns:
            bool: True if valve has been turned
        """
        # hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]
        handle_qpos = self.sim.data.qpos[self.handle_qpos_addr]
        return handle_qpos > 0.3 * np.pi

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the door handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the door handle
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.valve.important_sites["handle"],
                target_type="site"
            )

    # manipulation_env

    @property
    def _visualizations(self):
        """
        Visualization keywords for this environment
        Returns:
            set: All components that can be individually visualized for this environment
        """
        vis_set = super()._visualizations
        vis_set.add("grippers")
        return vis_set

    def _check_grasp(self, gripper, object_geoms):
        """
        Checks whether the specified gripper as defined by @gripper is grasping the specified object in the environment.
        By default, this will return True if at least one geom in both the "left_fingerpad" and "right_fingerpad" geom
        groups are in contact with any geom specified by @object_geoms. Custom gripper geom groups can be
        specified with @gripper as well.
        Args:
            gripper (GripperModel or str or list of str or list of list of str): If a MujocoModel, this is specific
            gripper to check for grasping (as defined by "left_fingerpad" and "right_fingerpad" geom groups). Otherwise,
                this sets custom gripper geom groups which together define a grasp. This can be a string
                (one group of single gripper geom), a list of string (multiple groups of single gripper geoms) or a
                list of list of string (multiple groups of multiple gripper geoms). At least one geom from each group
                must be in contact with any geom in @object_geoms for this method to return True.
            object_geoms (str or list of str or MujocoModel): If a MujocoModel is inputted, will check for any
                collisions with the model's contact_geoms. Otherwise, this should be specific geom name(s) composing
                the object to check for contact.
        Returns:
            bool: True if the gripper is grasping the given object
        """
        # Convert object, gripper geoms into standardized form
        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        else:
            o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
        if isinstance(gripper, GripperModel):
            g_geoms = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        elif type(gripper) is str:
            g_geoms = [[gripper]]
        else:
            # Parse each element in the gripper_geoms list accordingly
            g_geoms = [[g_group] if type(g_group) is str else g_group for g_group in gripper]

        # Search for collisions between each gripper geom group and the object geoms group
        for g_group in g_geoms:
            if not self.check_contact(g_group, o_geoms):
                return False
        return True

    def _gripper_to_target(self, gripper, target, target_type="body", return_distance=False):
        """
        Calculates the (x,y,z) Cartesian distance (target_pos - gripper_pos) from the specified @gripper to the
        specified @target. If @return_distance is set, will return the Euclidean (scalar) distance instead.
        Args:
            gripper (MujocoModel): Gripper model to update grip site rgb
            target (MujocoModel or str): Either a site / geom / body name, or a model that serves as the target.
                If a model is given, then the root body will be used as the target.
            target_type (str): One of {"body", "geom", or "site"}, corresponding to the type of element @target
                refers to.
            return_distance (bool): If set, will return Euclidean distance instead of Cartesian distance
        Returns:
            np.array or float: (Cartesian or Euclidean) distance from gripper to target
        """
        # Get gripper and target positions
        gripper_pos = self.sim.data.get_site_xpos(gripper.important_sites["grip_site"])
        # If target is MujocoModel, grab the correct body as the target and find the target position
        if isinstance(target, MujocoModel):
            target_pos = self.sim.data.get_body_xpos(target.root_body)
        elif target_type == "body":
            target_pos = self.sim.data.get_body_xpos(target)
        elif target_type == "site":
            target_pos = self.sim.data.get_site_xpos(target)
        else:
            target_pos = self.sim.data.get_geom_xpos(target)
        # Calculate distance
        diff = target_pos - gripper_pos
        # Return appropriate value
        return np.linalg.norm(diff) if return_distance else diff

    def _visualize_gripper_to_target(self, gripper, target, target_type="body"):
        """
        Colors the grip visualization site proportional to the Euclidean distance to the specified @target.
        Colors go from red --> green as the gripper gets closer.
        Args:
            gripper (MujocoModel): Gripper model to update grip site rgb
            target (MujocoModel or str): Either a site / geom / body name, or a model that serves as the target.
                If a model is given, then the root body will be used as the target.
            target_type (str): One of {"body", "geom", or "site"}, corresponding to the type of element @target
                refers to.
        """
        # Get gripper and target positions
        gripper_pos = self.sim.data.get_site_xpos(gripper.important_sites["grip_site"])
        # If target is MujocoModel, grab the correct body as the target and find the target position
        if isinstance(target, MujocoModel):
            target_pos = self.sim.data.get_body_xpos(target.root_body)
        elif target_type == "body":
            target_pos = self.sim.data.get_body_xpos(target)
        elif target_type == "site":
            target_pos = self.sim.data.get_site_xpos(target)
        else:
            target_pos = self.sim.data.get_geom_xpos(target)
        # color the gripper site appropriately based on (squared) distance to target
        dist = np.sum(np.square((target_pos - gripper_pos)))
        max_dist = 0.1
        scaled = (1.0 - min(dist / max_dist, 1.)) ** 15
        rgba = np.zeros(3)
        rgba[0] = 1 - scaled
        rgba[1] = scaled
        self.sim.model.site_rgba[self.sim.model.site_name2id(gripper.important_sites["grip_site"])][:3] = rgba

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure inputted robots and the corresponding requested task/configuration combo is legal.
        Should be implemented in every specific task module
        Args:
            robots (str or list of str): Inputted requested robots at the task-level environment
        """
        # Make sure all inputted robots are a manipulation robot
        if type(robots) is str:
            robots = [robots]
        for robot in robots:
            assert issubclass(ROBOT_CLASS_MAPPING[robot], Manipulator),\
                "Only manipulator robots supported for manipulation environment!"    


    # single_arm_env.py

    @property
    def _eef_xpos(self):
        """
        Grabs End Effector position

        Returns:
            np.array: End effector(x,y,z)
        """
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])

    @property
    def _eef_xmat(self):
        """
        End Effector orientation as a rotation matrix
        Note that this draws the orientation from the "ee" site, NOT the gripper site, since the gripper
        orientations are inconsistent!

        Returns:
            np.array: (3,3) End Effector orientation matrix
        """
        pf = self.robots[0].robot_model.naming_prefix
        if self.env_configuration == "bimanual":
            return np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(pf + "right_ee")]).reshape(3, 3)
        else:
            return np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(pf + "ee")]).reshape(3, 3)

    @property
    def _eef_xquat(self):
        """
        End Effector orientation as a (x,y,z,w) quaternion
        Note that this draws the orientation from the "ee" site, NOT the gripper site, since the gripper
        orientations are inconsistent!

        Returns:
            np.array: (x,y,z,w) End Effector quaternion
        """
        return mat2quat(self._eef_xmat)

    # valve
    @property
    def _handle_xpos(self):
        """
        Grabs the position of the valve handle.

        Returns:
            np.array: Valve handle (x,y,z)
        """
        return self.sim.data.site_xpos[self.valve_handle_site_id]

    @property
    def _gripper_to_handle(self):
        """
        Calculates distance from the gripper to the valve handle.

        Returns:
            np.array: (x,y,z) distance between handle and eef
        """
        return self._handle_xpos - self._eef_xpos
