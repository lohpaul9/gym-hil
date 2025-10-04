#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from gym_hil.controllers import opspace
from gym_hil.controllers.ik_control import ik_control

MAX_GRIPPER_COMMAND = 255


@dataclass(frozen=True)
class GymRenderingSpec:
    height: int = 128
    width: int = 128
    camera_id: str | int = -1
    mode: Literal["rgb_array", "human"] = "rgb_array"


class MujocoGymEnv(gym.Env):
    """MujocoEnv with gym interface."""

    def __init__(
        self,
        xml_path: Path,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
    ):
        self._model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        self._model.vis.global_.offwidth = render_spec.width
        self._model.vis.global_.offheight = render_spec.height
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = physics_dt
        self._control_dt = control_dt
        self._n_substeps = int(control_dt // physics_dt)
        self._random = np.random.RandomState(seed)
        self._viewer: Optional[mujoco.Renderer] = None
        self._render_specs = render_spec

    def render(self):
        if self._viewer is None:
            self._viewer = mujoco.Renderer(
                model=self._model,
                height=self._render_specs.height,
                width=self._render_specs.width,
            )
        self._viewer.update_scene(self._data, camera=self._render_specs.camera_id)
        return self._viewer.render()

    def close(self) -> None:
        """Release graphics resources if they exist.

        In MuJoCo < 2.3.0 `mujoco.Renderer` had no `close()` member.  Calling
        it unconditionally therefore raises `AttributeError`.  We check for
        the attribute first and fall back to a no-op, keeping compatibility
        across MuJoCo versions.
        """

        viewer = self._viewer
        if viewer is None:
            return

        if hasattr(viewer, "close") and callable(viewer.close):
            try:  # noqa: SIM105
                viewer.close()
            except Exception:
                # Ignore errors coming from already freed OpenGL contexts or
                # older MuJoCo builds.
                pass

        self._viewer = None

    # Accessors.

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def control_dt(self) -> float:
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        return self._model.opt.timestep

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random


class RobotGymEnv(MujocoGymEnv, ABC):
    """Abstract base class for robot-specific gym environments.

    Provides common interface for different robot types while handling
    robot-specific configuration like joint IDs, sensors, and control.
    """

    def __init__(
        self,
        xml_path: Path,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
    ):
        super().__init__(
            xml_path=xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.image_obs = image_obs

        # Setup robot-specific components (implemented by subclasses)
        self._setup_robot_ids()
        self._setup_cameras()
        self._setup_observation_space()
        self._setup_action_space()

        # Initialize renderer
        self._viewer = mujoco.Renderer(self.model, height=render_spec.height, width=render_spec.width)
        self._viewer.render()

    @abstractmethod
    def _setup_robot_ids(self):
        """Setup robot-specific joint and actuator IDs.

        Should cache joint IDs, actuator IDs, gripper IDs, and site IDs.
        """
        pass

    @abstractmethod
    def _setup_cameras(self):
        """Setup robot-specific camera IDs.

        Should set self.camera_id to a tuple of camera IDs.
        """
        pass

    @abstractmethod
    def _setup_observation_space(self):
        """Setup robot-specific observation space."""
        pass

    @abstractmethod
    def _setup_action_space(self):
        """Setup robot-specific action space."""
        pass

    @abstractmethod
    def reset_robot(self):
        """Reset the robot to its home position."""
        pass

    @abstractmethod
    def apply_action(self, action):
        """Apply action to the robot."""
        pass

    @abstractmethod
    def get_robot_state(self):
        """Get the current state of the robot."""
        pass

    @abstractmethod
    def get_gripper_pose(self):
        """Get the current pose/state of the gripper."""
        pass

    def render(self):
        """Render the environment and return frames from multiple cameras."""
        rendered_frames = []
        for cam_id in self.camera_id:
            self._viewer.update_scene(self.data, camera=cam_id)
            rendered_frames.append(self._viewer.render())
        return rendered_frames


class FrankaGymEnv(RobotGymEnv):
    """Base class for Franka Panda robot environments."""

    def __init__(
        self,
        xml_path: Path | None = None,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        home_position: np.ndarray = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4)),  # noqa: B008
        cartesian_bounds: np.ndarray = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]]),  # noqa: B008
    ):
        if xml_path is None:
            xml_path = Path(__file__).parent.parent / "gym_hil" / "assets" / "scene.xml"

        self._home_position = home_position
        self._cartesian_bounds = cartesian_bounds

        super().__init__(
            xml_path=xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
        )

    def _setup_robot_ids(self):
        """Setup Franka Panda robot-specific joint and actuator IDs."""
        self._panda_dof_ids = np.asarray([self._model.joint(f"joint{i}").id for i in range(1, 8)])
        self._panda_ctrl_ids = np.asarray([self._model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id

    def _setup_cameras(self):
        """Setup Franka Panda robot-specific cameras."""
        camera_name_1 = "front"
        camera_name_2 = "handcam_rgb"
        camera_id_1 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_1)
        camera_id_2 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_2)
        self.camera_id = (camera_id_1, camera_id_2)

    def _setup_observation_space(self):
        """Setup the observation space for the Franka environment."""
        base_obs_space = {
            "agent_pos": spaces.Dict(
                {
                    "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                    "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                    "gripper_pose": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
                }
            )
        }

        self.observation_space = spaces.Dict(base_obs_space)

        if self.image_obs:
            self.observation_space = spaces.Dict(
                {
                    **base_obs_space,
                    "pixels": spaces.Dict(
                        {
                            "front": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                }
            )

    def _setup_action_space(self):
        """Setup the action space for the Franka environment."""
        self.action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset_robot(self):
        """Reset the robot to home position."""
        self._data.qpos[self._panda_dof_ids] = self._home_position
        self._data.ctrl[self._panda_ctrl_ids] = 0.0
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        self._data.mocap_pos[0] = tcp_pos

    def apply_action(self, action):
        """Apply the action to the robot."""
        x, y, z, rx, ry, rz, grasp_command = action

        # Set the mocap position
        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z])
        npos = np.clip(pos + dpos, *self._cartesian_bounds)
        self._data.mocap_pos[0] = npos

        # Set gripper grasp
        g = self._data.ctrl[self._gripper_ctrl_id] / MAX_GRIPPER_COMMAND
        ng = np.clip(g + grasp_command, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_id] = ng * MAX_GRIPPER_COMMAND

        # Apply operational space control
        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=self._home_position,
                gravity_comp=True,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

    def get_robot_state(self):
        """Get the current state of the robot."""
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        # tcp_quat = self._data.sensor("2f85/pinch_quat").data
        # tcp_vel = self._data.sensor("2f85/pinch_vel").data
        # tcp_angvel = self._data.sensor("2f85/pinch_angvel").data
        qpos = self.data.qpos[self._panda_dof_ids].astype(np.float32)
        qvel = self.data.qvel[self._panda_dof_ids].astype(np.float32)
        gripper_pose = self.get_gripper_pose()

        return np.concatenate([qpos, qvel, gripper_pose, tcp_pos])

    def get_gripper_pose(self):
        """Get the current pose of the gripper."""
        return np.array([self._data.ctrl[self._gripper_ctrl_id]], dtype=np.float32)


class SO101GymEnv(RobotGymEnv):
    """Base class for SO-101 robot environments."""

    def __init__(
        self,
        xml_path: Path | None = None,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        home_position: np.ndarray = np.asarray((0.0, 0.0, 0.0, 0.0, 0.0)),  # noqa: B008
        cartesian_bounds: np.ndarray = np.asarray([[0.1, -0.3, 0.0], [0.5, 0.3, 0.4]]),  # noqa: B008
    ):
        if xml_path is None:
            xml_path = Path(__file__).parent.parent / "gym_hil" / "assets" / "SO101" / "pick_scene.xml"

        self._home_position = home_position
        self._cartesian_bounds = cartesian_bounds

        super().__init__(
            xml_path=xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
        )

    def _setup_robot_ids(self):
        """Setup SO-101 robot-specific joint and actuator IDs."""
        # SO-101 has 5 arm joints + 1 gripper
        self._arm_dof_ids = np.asarray([self._model.joint(name).id for name in
                                        ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]])
        self._arm_ctrl_ids = np.asarray([self._model.actuator(name).id for name in
                                         ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]])
        self._gripper_ctrl_id = self._model.actuator("gripper").id
        self._ee_site_id = self._model.site("gripperframe").id

        # Cached target position for stable IK tracking
        self._target_ee_pos = None
        self._last_action_was_zero = False

    def _setup_cameras(self):
        """Setup SO-101 robot-specific cameras."""
        camera_name = "front"
        camera_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.camera_id = (camera_id,)  # Tuple with single camera

    def _setup_observation_space(self):
        """Setup the observation space for the SO-101 environment."""
        base_obs_space = {
            "agent_pos": spaces.Dict(
                {
                    "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                    "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                    "gripper_pose": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
                }
            )
        }

        self.observation_space = spaces.Dict(base_obs_space)

        if self.image_obs:
            self.observation_space = spaces.Dict(
                {
                    **base_obs_space,
                    "pixels": spaces.Dict(
                        {
                            "front": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                }
            )

    def _setup_action_space(self):
        """Setup the action space for the SO-101 environment."""
        # 7D action space: [dx, dy, dz, drx, dry, drz, gripper]
        # Note: rotation deltas will be ignored since SO-101 only has 5 DOF
        self.action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset_robot(self):
        """Reset the robot to home position."""
        self._data.qpos[self._arm_dof_ids] = self._home_position
        self._data.ctrl[self._arm_ctrl_ids] = 0.0
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position and orientation
        ee_pos = self._data.sensor("so101/ee_pos").data
        ee_quat = self._data.sensor("so101/ee_quat").data
        self._data.mocap_pos[0] = ee_pos
        self._data.mocap_quat[0] = ee_quat

        # Initialize cached target position
        self._target_ee_pos = ee_pos.copy()
        self._last_action_was_zero = False

    def apply_action(self, action):
        """Apply the action to the robot using position-based IK control."""
        x, y, z, rx, ry, rz, grasp_command = action

        # Accumulate target position from delta
        # Action is a position delta to add to the current target
        delta = np.asarray([x, y, z])
        action_is_zero = np.linalg.norm(delta) < 1e-6

        # On transition from motion to hold (first zero action), lock target to current position
        if action_is_zero and not self._last_action_was_zero:
            current_ee_pos = self._data.site_xpos[self._ee_site_id].copy()
            self._target_ee_pos = current_ee_pos
        elif not action_is_zero:
            # Normal motion: accumulate delta
            self._target_ee_pos += delta

        # Track action state for next step
        self._last_action_was_zero = action_is_zero

        # Clip to workspace bounds if defined
        if hasattr(self, '_cartesian_bounds') and self._cartesian_bounds is not None:
            self._target_ee_pos = np.clip(
                self._target_ee_pos,
                self._cartesian_bounds[0],
                self._cartesian_bounds[1]
            )

        # Set gripper - SO-101 gripper uses position control
        # Normalize current position to [0, 1], apply delta, scale back
        gripper_range = self._model.actuator("gripper").ctrlrange
        current_gripper = self._data.ctrl[self._gripper_ctrl_id]
        # Normalize to [0, 1]
        g_norm = (current_gripper - gripper_range[0]) / (gripper_range[1] - gripper_range[0])
        # Apply command as delta in [-1, 1]
        ng_norm = np.clip(g_norm + grasp_command, 0.0, 1.0)
        # Scale back to control range
        new_gripper = gripper_range[0] + ng_norm * (gripper_range[1] - gripper_range[0])
        self._data.ctrl[self._gripper_ctrl_id] = new_gripper

        # Apply position-based IK control to reach target
        for _ in range(self._n_substeps):
            tau = ik_control(
                model=self._model,
                data=self._data,
                site_id=self._ee_site_id,
                dof_ids=self._arm_dof_ids,
                target_pos=self._target_ee_pos,
                joint_kp=500.0,
                joint_kd=50.0,
                ik_method="levenberg_marquardt",
                ik_damping=0.1,
                ik_iterations=20,
                gravity_comp=True,
            )
            self._data.ctrl[self._arm_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

    def get_robot_state(self):
        """Get the current state of the robot."""
        ee_pos = self._data.sensor("so101/ee_pos").data
        qpos = self.data.qpos[self._arm_dof_ids].astype(np.float32)
        qvel = self.data.qvel[self._arm_dof_ids].astype(np.float32)
        gripper_pose = self.get_gripper_pose()

        return np.concatenate([qpos, qvel, gripper_pose, ee_pos])

    def get_gripper_pose(self):
        """Get the current pose of the gripper."""
        # Normalize gripper position to [-1, 1] range
        gripper_range = self._model.actuator("gripper").ctrlrange
        gripper_pos = self._data.ctrl[self._gripper_ctrl_id]
        normalized = 2 * (gripper_pos - gripper_range[0]) / (gripper_range[1] - gripper_range[0]) - 1
        return np.array([normalized], dtype=np.float32)

    def render(self):
        """Render the environment and return frame from front camera."""
        # SO-101 only has one camera, so return single frame (not list)
        self._viewer.update_scene(self.data, camera=self.camera_id[0])
        return self._viewer.render()
