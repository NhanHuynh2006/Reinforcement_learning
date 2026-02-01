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
from nav_msgs.msg import Odometry

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


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


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
        self.odom_sub = self.create_subscription(Odometry, "/diff_cont/odom", self.odom_cb, 10)
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

        self.off_lane_thresh = 0.9
        self.off_lane_penalty = 1.0
        self.collision_penalty = 5.0
        self.forward_reward_k = 0.35
        self.min_forward_frac = 0.05
        self.low_speed_penalty = 0.2
        self.angular_penalty_k = 0.1
        self.smooth_penalty_k = 0.02
        self.off_lane_grace = 10
        self.off_lane_count = 0
        self.progress_reward_k = 3.0
        self.back_progress_penalty_k = 3.0
        self.no_progress_thresh = 0.05
        self.no_progress_penalty = 0.3
        self.spin_penalty_k = 0.7
        self.no_progress_grace = 25
        self.no_progress_count = 0
        self.terminate_on_no_progress = True
        self.reverse_progress_thresh = -0.03
        self.reverse_progress_grace = 10
        self.reverse_progress_count = 0
        self.reverse_action_grace = 15
        self.reverse_action_count = 0
        self.stuck_progress_thresh = 0.05
        self.stuck_grace = 20
        self.stuck_count = 0
        self.spin_w_thresh = 0.6 * self.max_w
        self.spin_grace = 15
        self.spin_count = 0
        self.missing_mask_grace = 8
        self.missing_mask_count = 0
        self.terminate_on_missing = True
        self.allow_reverse = False
        self.back_eps = 0.1 * self.max_v
        self.back_penalty_k = 0.4
        self.center_floor = 0.25
        self.roi_y_start = 0.6
        # Scale thresholds with obs size to avoid "missing" when obs_size changes
        self.min_road_width_px = max(10, int(self.obs_size * 0.12))
        self.max_road_width_px = int(self.obs_size * 0.98)
        self.min_lane_width_px = max(6, int(self.obs_size * 0.05))
        self.max_lane_width_px = int(self.obs_size * 0.8)
        self.center_pos_min = 0.0
        self.center_pos_max = 1.0

        self.model_positions: Dict[str, Tuple[float, float]] = {}
        self.odom_pos = None
        self.odom_yaw = None
        self.have_odom = False
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_pos = None
        self.debug_pub = None
        if self.publish_debug:
            self.debug_pub = self.create_publisher(Image, self.debug_topic, 10)
        self._last_obs = None
        # REWARD LOG (toggle + frequency)
        self.log_reward_parts = True
        self.reward_log_every = 1
        self.reward_log_use_print = True
        self.log_model_names = True

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

    def odom_cb(self, msg: Odometry) -> None:
        self.odom_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.odom_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.have_odom = True

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
        yellow_mask = obs[y0:, :, 1] > 0
        white_mask = obs[y0:, :, 2] > 0

        if not yellow_mask.any():
            return None, True

        y_cols = np.where(yellow_mask)[1]
        y_center = float(np.mean(y_cols))

        w_cols = np.where(white_mask)[1]
        w_cols = w_cols[w_cols > y_center + self.min_lane_width_px]
        if w_cols.size == 0:
            return None, True

        w_center = float(np.mean(w_cols))
        lane_width = w_center - y_center
        if lane_width < self.min_lane_width_px or lane_width > self.max_lane_width_px:
            return None, True

        lane_center = y_center + 0.5 * lane_width
        img_center = w / 2.0
        deviation = abs(lane_center - img_center) / img_center
        pos = (img_center - y_center) / max(lane_width, 1e-6)
        if pos < self.center_pos_min or pos > self.center_pos_max:
            return deviation, True
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
        raw_v = float(np.clip(action[0], -1.0, 1.0) * self.max_v)
        w = float(np.clip(action[1], -1.0, 1.0) * self.max_w)
        if self.allow_reverse:
            v = raw_v
        else:
            v = max(0.0, raw_v)

        self._publish_cmd(v, w)

        t_end = time.time() + self.step_dt
        while rclpy.ok() and time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.0)

        obs = self._get_obs()
        if self.show_debug or self.publish_debug:
            self._debug_output(obs)
        deviation, off_lane = self._road_deviation(obs)
        if deviation is None:
            deviation, off_lane = self._lane_deviation(obs)

        reward = 0.0
        terminated = False
        r_center = 0.0
        r_missing = 0.0
        r_off_lane = 0.0
        r_forward = 0.0
        r_low_speed = 0.0
        r_back = 0.0
        r_angular = 0.0
        r_smooth = 0.0
        r_progress = 0.0
        r_back_prog = 0.0

        pos = self.odom_pos if self.have_odom else self.model_positions.get(self.model_name)
        progress = None
        dist = None
        if pos is not None and self.prev_pos is not None:
            dx = pos[0] - self.prev_pos[0]
            dy = pos[1] - self.prev_pos[1]
            dist = math.hypot(dx, dy)
            norm = max(self.max_v * self.step_dt, 1e-6)
            if self.have_odom and self.odom_yaw is not None:
                forward = dx * math.cos(self.odom_yaw) + dy * math.sin(self.odom_yaw)
                progress = forward / norm
            else:
                progress = dist / norm
            progress = max(-2.0, min(2.0, progress))

        if deviation is None:
            reward -= 1.0
            r_missing -= 1.0
            self.missing_mask_count += 1
        else:
            self.missing_mask_count = 0
            r_center = 0.5 * (1.0 - deviation)
            reward += r_center
            if off_lane:
                reward -= self.off_lane_penalty
                r_off_lane -= self.off_lane_penalty
                self.off_lane_count += 1
            else:
                self.off_lane_count = 0

        # Grace period to avoid instant reset on noisy masks
        if self.off_lane_count >= self.off_lane_grace:
            terminated = True
        if self.terminate_on_missing and self.missing_mask_count >= self.missing_mask_grace:
            terminated = True

        center_factor = 0.0
        if deviation is not None:
            center_factor = max(0.0, min(1.0, 1.0 - deviation))
            if off_lane:
                center_factor = 0.0

        # Forward reward + discourage stopping (scaled by center confidence)
        if deviation is not None:
            if v > 0.0:
                r_forward = self.forward_reward_k * (v / max(self.max_v, 1e-6)) * center_factor
                reward += r_forward
            if v < self.min_forward_frac * self.max_v:
                reward -= self.low_speed_penalty
                r_low_speed -= self.low_speed_penalty

        # Penalize reversing
        if raw_v < -self.back_eps:
            r_back = -self.back_penalty_k * (abs(raw_v) / max(self.max_v, 1e-6))
            reward += r_back
            self.reverse_action_count += 1
            if self.reverse_action_count >= self.reverse_action_grace:
                terminated = True
        else:
            self.reverse_action_count = 0

        # Penalize excessive steering and jitter only when we have a valid mask
        if deviation is not None:
            r_angular = -self.angular_penalty_k * (abs(w) / max(self.max_w, 1e-6))
            reward += r_angular
            delta = np.abs(np.array([v, w]) - self.prev_action)
            delta_norm = (delta[0] / max(self.max_v, 1e-6)) + (delta[1] / max(self.max_w, 1e-6))
            r_smooth = -self.smooth_penalty_k * delta_norm
            reward += r_smooth
        self.prev_action = np.array([v, w], dtype=np.float32)

        if progress is not None:
            r_progress = self.progress_reward_k * progress * center_factor
            reward += r_progress
            if progress < self.reverse_progress_thresh:
                r_back_prog = -self.back_progress_penalty_k * abs(progress)
                reward += r_back_prog
                self.reverse_progress_count += 1
                if self.reverse_progress_count >= self.reverse_progress_grace:
                    terminated = True
            else:
                self.reverse_progress_count = 0
            if progress < self.no_progress_thresh:
                self.no_progress_count += 1
                reward -= self.no_progress_penalty
                reward -= self.spin_penalty_k * (abs(w) / max(self.max_w, 1e-6))
                if self.terminate_on_no_progress and self.no_progress_count >= self.no_progress_grace:
                    terminated = True
            else:
                self.no_progress_count = 0
            if progress < self.stuck_progress_thresh and center_factor > 0.2:
                self.stuck_count += 1
                if abs(w) > self.spin_w_thresh:
                    self.spin_count += 1
                else:
                    self.spin_count = 0
                if self.stuck_count >= self.stuck_grace or self.spin_count >= self.spin_grace:
                    terminated = True
            else:
                self.stuck_count = 0
                self.spin_count = 0
        else:
            # Fallback: reward forward command if we don't have position info
            if v > 0.0 and deviation is not None:
                r_progress = self.progress_reward_k * (v / max(self.max_v, 1e-6)) * center_factor
                reward += r_progress
            self.reverse_progress_count = 0
            self.stuck_count = 0
            self.spin_count = 0

        if self._check_collision():
            reward -= self.collision_penalty
            terminated = True

        if pos is not None:
            self.prev_pos = (pos[0], pos[1])

        if self.log_reward_parts and self.reward_log_every > 0 and (self.step_count % self.reward_log_every) == 0:
            msg = (
                "Reward parts: "
                f"center={r_center:.3f} missing={r_missing:.3f} off_lane={r_off_lane:.3f} "
                f"forward={r_forward:.3f} low_speed={r_low_speed:.3f} back={r_back:.3f} "
                f"angular={r_angular:.3f} smooth={r_smooth:.3f} "
                f"progress={progress if progress is not None else 0.0:.6f} "
                f"center_fac={center_factor:.3f} prog_r={r_progress:.3f} back_prog={r_back_prog:.3f} "
                f"total={reward:.3f}"
            )
            self.get_logger().info(msg)
            if self.reward_log_use_print:
                print(msg, flush=True)
            if pos is not None:
                pos_msg = (
                    f"POS: x={pos[0]:.4f} y={pos[1]:.4f} "
                    f"prev={self.prev_pos[0]:.4f},{self.prev_pos[1]:.4f} "
                    f"dist={dist if dist is not None else 0.0:.6f} "
                    f"progress={progress if progress is not None else 0.0:.6f}"
                )
                self.get_logger().info(pos_msg)
                if self.reward_log_use_print:
                    print(pos_msg, flush=True)
            elif self.log_model_names:
                names = list(self.model_positions.keys())
                names_msg = f"Model names: {names[:8]} odom={self.have_odom}"
                self.get_logger().info(names_msg)
                if self.reward_log_use_print:
                    print(names_msg, flush=True)

        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        info = {
            "v": v,
            "w": w,
            "deviation": deviation if deviation is not None else -1.0,
            "min_obstacle_dist": self.last_min_dist,
            "off_lane": off_lane,
        }

        return obs, float(reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.last_min_dist = float("inf")
        self.off_lane_count = 0
        self.no_progress_count = 0
        self.reverse_progress_count = 0
        self.reverse_action_count = 0
        self.stuck_count = 0
        self.spin_count = 0
        self.missing_mask_count = 0
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_pos = None

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
