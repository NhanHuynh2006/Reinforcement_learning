#!/usr/bin/env python3
import math
import time
from typing import Dict, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty
from sensor_msgs.msg import Image

import gymnasium as gym
from gymnasium import spaces

from segmentation import LaneSegmenter


def euler_to_quaternion(roll: float, pitch: float, yaw: float):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


class LaneFollowerSegEnv(gym.Env, Node):
    """RL env using segmentation masks as observations."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cam_topic: str = "/camera/image_raw",
        cmd_vel_topic: str = "/diff_cont/cmd_vel_unstamped",
        max_v: float = 0.5,
        max_w: float = 1.5,
        step_dt: float = 0.1,
        obs_size: int = 84,
        yolo_weights: str | None = None,
        yolo_conf: float = 0.25,
        yolo_iou: float = 0.45,
        yolo_imgsz: int = 320,
        yolo_device: str | None = None,
        yolo_every: int = 1,
        show_debug: bool = False,
        debug_every: int = 5,
        publish_debug: bool = False,
        debug_topic: str = "/segmentation/debug",
    ) -> None:
        if not rclpy.ok():
            rclpy.init(args=None)

        gym.Env.__init__(self)
        Node.__init__(self, "lane_follower_seg_env")

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, cam_topic, self.image_cb, 10)
        self.model_sub = self.create_subscription(ModelStates, "/gazebo/model_states", self.model_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        self.segmenter = LaneSegmenter(
            yolo_weights=yolo_weights,
            yolo_conf=yolo_conf,
            yolo_iou=yolo_iou,
            imgsz=yolo_imgsz,
            device=yolo_device,
            yolo_every=yolo_every,
        )
        self.obs_size = int(obs_size)
        self.observation_space = spaces.Box(0, 255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.latest_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.have_img = False

        self.max_v = float(max_v)
        self.max_w = float(max_w)
        self.step_dt = float(step_dt)
        self.step_count = 0
        self.max_steps = 2000
        self.show_debug = bool(show_debug)
        self.debug_every = max(1, int(debug_every))
        self.publish_debug = bool(publish_debug)
        self.debug_topic = debug_topic

        self.model_name = "my_bot"
        self.obstacle_names = {"cone_1", "cone_2", "block_1", "block_2"}
        self.collision_distance = 0.35
        self.last_min_dist = float("inf")

        self.init_pose_xyzrpy = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        self.cli_pause, self.srv_pause = self._pick_empty(["/pause_physics", "/gazebo/pause_physics"])
        self.cli_unpause, self.srv_unpause = self._pick_empty(["/unpause_physics", "/gazebo/unpause_physics"])
        self.cli_reset_world, self.srv_reset_world = self._pick_empty(["/reset_world", "/gazebo/reset_world"])
        self.cli_set_state, self.srv_set_state = self._pick_set_model_state(["/set_model_state", "/gazebo/set_model_state"])

        self.off_lane_thresh = 0.6
        self.off_lane_penalty = 3.0
        self.deviation_k = 1.5
        self.center_deadband = 0.15
        self.center_penalty_k = 1.5
        self.collision_penalty = 5.0
        self.forward_reward_k = 1.0
        self.min_forward_frac = 0.03
        self.low_speed_penalty = 0.05
        self.angular_penalty_k = 0.02
        self.smooth_penalty_k = 0.01
        self.off_lane_grace = 8
        self.off_lane_count = 0
        self.missing_mask_grace = 6
        self.missing_mask_count = 0
        self.terminate_on_missing_mask = True
        self.terminate_on_offlane = True
        self.wheel_separation = 0.35
        self.allow_reverse = False
        self.min_forward_v = 0.25
        self.action_smooth = 0.6
        self.angular_scale = 0.7
        self.back_eps = 0.1 * self.max_v
        self.back_penalty_k = 0.6
        self.reverse_both_penalty = 0.8
        self.progress_reward_k = 1.2
        self.stuck_dist_thresh = 0.01
        self.stuck_grace = 10
        self.stuck_penalty = 1.0
        self.roi_y_start = 0.6
        self.min_road_width_px = 20
        self.max_road_width_px = 84
        self.min_lane_width_ratio = 0.15
        self.max_lane_width_ratio = 0.95
        self.center_pos_min = 0.35
        self.center_pos_max = 0.65

        self.model_positions: Dict[str, Tuple[float, float]] = {}
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_pos = None
        self.stuck_count = 0
        self.debug_pub = None
        if self.publish_debug:
            self.debug_pub = self.create_publisher(Image, self.debug_topic, 10)
        self._last_obs = None

    def _pick_empty(self, names):
        for n in names:
            c = self.create_client(Empty, n)
            if c.wait_for_service(timeout_sec=0.5):
                return c, n
        return self.create_client(Empty, names[0]), names[0]

    def _pick_set_model_state(self, names):
        for n in names:
            c = self.create_client(SetModelState, n)
            if c.wait_for_service(timeout_sec=0.5):
                return c, n
        return None, None

    def image_cb(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_img = cv_image
            self.have_img = True
        except Exception as exc:
            self.get_logger().warn(f"Image convert error: {exc}")

    def model_cb(self, msg: ModelStates) -> None:
        positions = {}
        for name, pose in zip(msg.name, msg.pose):
            positions[name] = (pose.position.x, pose.position.y)
        self.model_positions = positions

    def _call_empty(self, client, name: str, timeout: float = 2.0) -> bool:
        if client is None:
            return False
        if not client.wait_for_service(timeout_sec=timeout):
            return False
        try:
            fut = client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
            return fut.result() is not None
        except Exception:
            return False

    def _teleport_to_start(self, timeout: float = 2.0) -> bool:
        if self.cli_set_state is None:
            return False
        if not self.cli_set_state.wait_for_service(timeout_sec=timeout):
            return False

        x, y, z, r, p, yaw = self.init_pose_xyzrpy
        qx, qy, qz, qw = euler_to_quaternion(r, p, yaw)

        state = ModelState()
        state.model_name = self.model_name
        state.pose = Pose()
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.x = qx
        state.pose.orientation.y = qy
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw

        try:
            req = SetModelState.Request()
            req.model_state = state
            fut = self.cli_set_state.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
            return fut.result() is not None
        except Exception:
            return False

    def _reset_world(self) -> None:
        self._call_empty(self.cli_pause, self.srv_pause, timeout=2.0)
        self._call_empty(self.cli_reset_world, self.srv_reset_world, timeout=2.0)
        self._publish_cmd(0.0, 0.0)
        if self.cli_set_state is not None:
            self._teleport_to_start(timeout=2.0)
        for _ in range(3):
            rclpy.spin_once(self, timeout_sec=0.05)
        self._call_empty(self.cli_unpause, self.srv_unpause, timeout=2.0)

    def _publish_cmd(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def _get_obs(self) -> np.ndarray:
        if not self.have_img:
            return np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        obs = self.segmenter.build_obs(self.latest_img, size=self.obs_size)
        self._last_obs = obs
        return obs

    def _debug_output(self, obs: np.ndarray) -> None:
        if (self.step_count % self.debug_every) != 0:
            return
        try:
            h, w, _ = obs.shape
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            road = obs[:, :, 0] > 0
            yellow = obs[:, :, 1] > 0
            white = obs[:, :, 2] > 0
            vis[road] = (0, 255, 0)
            vis[yellow] = (0, 255, 255)
            vis[white] = (255, 255, 255)

            if self.show_debug:
                cv2.imshow("seg_mask", vis)
                cv2.waitKey(1)
            if self.publish_debug and self.debug_pub is not None:
                msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
                self.debug_pub.publish(msg)
        except Exception:
            pass

    def _road_deviation(self, obs: np.ndarray) -> tuple[float | None, bool]:
        h, w, _ = obs.shape
        y0 = int(h * self.roi_y_start)
        road_mask = obs[y0:, :, 0] > 0

        if not road_mask.any():
            return None, True

        cols = np.where(road_mask)[1]
        min_c = float(np.min(cols))
        max_c = float(np.max(cols))
        road_width = max_c - min_c
        if road_width < self.min_road_width_px or road_width > self.max_road_width_px:
            return None, True

        road_center = 0.5 * (min_c + max_c)
        img_center = w / 2.0
        deviation = abs(road_center - img_center) / img_center
        off_lane = deviation > self.off_lane_thresh
        return deviation, off_lane

    def _lane_deviation(self, obs: np.ndarray) -> tuple[float | None, bool]:
        h, w, _ = obs.shape
        y0 = int(h * self.roi_y_start)
        white_mask = obs[y0:, :, 2] > 0

        if not white_mask.any():
            return None, True

        cols = np.where(white_mask)[1]
        left_cols = cols[cols < (w / 2.0)]
        right_cols = cols[cols > (w / 2.0)]
        if left_cols.size == 0 or right_cols.size == 0:
            return None, True

        left_center = float(np.mean(left_cols))
        right_center = float(np.mean(right_cols))
        lane_width = right_center - left_center
        min_w = self.min_lane_width_ratio * w
        max_w = self.max_lane_width_ratio * w
        if lane_width < min_w or lane_width > max_w:
            return None, True

        lane_center = 0.5 * (left_center + right_center)
        img_center = w / 2.0
        deviation = abs(lane_center - img_center) / img_center
        off_lane = deviation > self.off_lane_thresh
        return deviation, off_lane

    def _check_collision(self) -> bool:
        if self.model_name not in self.model_positions:
            return False
        rx, ry = self.model_positions[self.model_name]
        min_dist = float("inf")
        for name in self.obstacle_names:
            if name in self.model_positions:
                ox, oy = self.model_positions[name]
                d = math.hypot(rx - ox, ry - oy)
                min_dist = min(min_dist, d)
        self.last_min_dist = min_dist
        return min_dist < self.collision_distance

    def step(self, action):
        a0 = float(np.clip(action[0], -1.0, 1.0))
        a1 = float(np.clip(action[1], -1.0, 1.0))
        if self.allow_reverse:
            v_cmd = a0 * self.max_v
        else:
            v_cmd = self.min_forward_v + max(0.0, a0) * (self.max_v - self.min_forward_v)
        w_cmd = a1 * self.max_w * self.angular_scale

        # Smooth commands to reduce jitter
        v = (1.0 - self.action_smooth) * self.prev_action[0] + self.action_smooth * v_cmd
        w = (1.0 - self.action_smooth) * self.prev_action[1] + self.action_smooth * w_cmd
        if not self.allow_reverse:
            v = max(self.min_forward_v, v)

        self._publish_cmd(v, w)

        t_end = time.time() + self.step_dt
        while rclpy.ok() and time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.0)

        obs = self._get_obs()
        if self.show_debug or self.publish_debug:
            self._debug_output(obs)
        deviation, off_lane = self._lane_deviation(obs)

        reward = 0.0
        terminated = False
        term_reason = ""

        if deviation is None:
            reward -= 1.0
            self.missing_mask_count += 1
        else:
            self.missing_mask_count = 0
            reward += 1.0 - self.deviation_k * deviation
            if deviation > self.center_deadband:
                norm = (deviation - self.center_deadband) / max(1.0 - self.center_deadband, 1e-6)
                reward -= self.center_penalty_k * norm
            if off_lane:
                reward -= self.off_lane_penalty
                self.off_lane_count += 1
                if self.terminate_on_offlane and self.off_lane_count >= self.off_lane_grace:
                    terminated = True
                    term_reason = "off_lane"
            else:
                self.off_lane_count = 0

        # Stop only if we lose the mask too long
        if self.missing_mask_count >= self.missing_mask_grace:
            if self.terminate_on_missing_mask:
                terminated = True
                term_reason = "missing_mask"
            else:
                reward -= 1.0

        # Forward reward + discourage stopping
        if v > 0.0:
            reward += self.forward_reward_k * (v / max(self.max_v, 1e-6))
        if v < self.min_forward_frac * self.max_v:
            reward -= self.low_speed_penalty

        # Penalize reversing
        if v < -self.back_eps:
            reward -= self.back_penalty_k * (abs(v) / max(self.max_v, 1e-6))
        # Extra penalty when both wheels are reversing
        wl = v - w * (self.wheel_separation / 2.0)
        wr = v + w * (self.wheel_separation / 2.0)
        if wl < -self.back_eps and wr < -self.back_eps:
            reward -= self.reverse_both_penalty

        # Penalize excessive steering and jitter
        reward -= self.angular_penalty_k * (abs(w) / max(self.max_w, 1e-6))
        delta = np.abs(np.array([v, w]) - self.prev_action)
        delta_norm = (delta[0] / max(self.max_v, 1e-6)) + (delta[1] / max(self.max_w, 1e-6))
        reward -= self.smooth_penalty_k * delta_norm
        self.prev_action = np.array([v, w], dtype=np.float32)

        # Progress reward + stuck penalty
        pos = self.model_positions.get(self.model_name)
        if pos is not None:
            if self.prev_pos is not None:
                dx = pos[0] - self.prev_pos[0]
                dy = pos[1] - self.prev_pos[1]
                dist = math.hypot(dx, dy)
                norm = max(self.max_v * self.step_dt, 1e-6)
                reward += self.progress_reward_k * (dist / norm)
                if dist < self.stuck_dist_thresh:
                    self.stuck_count += 1
                else:
                    self.stuck_count = 0
                if self.stuck_count >= self.stuck_grace:
                    reward -= self.stuck_penalty
            self.prev_pos = (pos[0], pos[1])


        if self._check_collision():
            reward -= self.collision_penalty
            terminated = True
            term_reason = "collision"

        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        info = {
            "v": v,
            "w": w,
            "deviation": deviation if deviation is not None else -1.0,
            "min_obstacle_dist": self.last_min_dist,
            "off_lane": off_lane,
            "term_reason": term_reason,
        }

        return obs, float(reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.last_min_dist = float("inf")
        self.off_lane_count = 0
        self.missing_mask_count = 0
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_pos = None
        self.stuck_count = 0

        self._publish_cmd(0.0, 0.0)
        self._reset_world()

        self.have_img = False
        t0 = time.time()
        while rclpy.ok() and not self.have_img and time.time() - t0 < 2.0:
            rclpy.spin_once(self, timeout_sec=0.1)

        return self._get_obs(), {}

    def close(self):
        try:
            self._publish_cmd(0.0, 0.0)
        except Exception:
            pass
        try:
            if self.show_debug:
                cv2.destroyAllWindows()
        except Exception:
            pass
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    env = LaneFollowerSegEnv()
    try:
        obs, info = env.reset()
        print("Reset done. Obs shape:", obs.shape)
        for i in range(20):
            a = env.action_space.sample()
            obs, r, term, trunc, inf = env.step(a)
            print(
                f"step={i:02d} r={r:.3f} term={term} trunc={trunc} v={inf['v']:.2f} w={inf['w']:.2f}"
            )
            if term or trunc:
                obs, info = env.reset()
    finally:
        env.close()
