#!/usr/bin/env python3
import time
import math
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty

import gymnasium as gym
from gymnasium import spaces

def euler_to_quaternion(roll: float, pitch: float, yaw: float):
    """Convert Euler angles (rad) to quaternion (x, y, z, w)."""
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


class LaneFollowerEnv(gym.Env, Node):
    """
    Gymnasium Env điều khiển robot vi sai trong Gazebo Classic bằng ROS 2.
    - Reset: ưu tiên /reset_world (root); nếu có /set_model_state thì teleport về pose cấu hình.
    - Action: [wl, wr] chuẩn hóa [-1, 1] -> rad/s.
    - Obs: ảnh 84x84x3 (BGR).
    - Reward: 1 - deviation của vạch vàng so với tâm ảnh; mất vạch -> -1 và terminate.
    """

    metadata = {"render_modes": []}

    # -------------------- ctor --------------------
    def __init__(self,
                 left_topic='/left_wheel_vel/commands',
                 right_topic='/right_wheel_vel/commands',
                 cam_topic='/camera/image_raw',
                 max_wheel_rad_s=8.0,
                 step_dt=0.05):
        # Init ROS (nếu chạy độc lập)
        if not rclpy.ok():
            rclpy.init(args=None)

        gym.Env.__init__(self)
        Node.__init__(self, 'lane_follower_env')

        # ------------ ROS pubs/subs ------------
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, cam_topic, self.image_cb, 10)
        self.left_pub  = self.create_publisher(Float64MultiArray, left_topic, 10)
        self.right_pub = self.create_publisher(Float64MultiArray, right_topic, 10)

        # ------------ Spaces ------------
        self.obs_shape = (84, 84, 3)
        self.observation_space = spaces.Box(0, 255, shape=self.obs_shape, dtype=np.uint8)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # ------------ Runtime state ------------
        self.latest_img = np.zeros(self.obs_shape, dtype=np.uint8)
        self.have_img = False
        self.max_w = float(max_wheel_rad_s)
        self.step_dt = float(step_dt)
        self.step_count = 0
        self.max_steps = 2000

        # ------------ Robot/model settings ------------
        self.model_name = 'my_bot'  # <-- ĐÚNG tên --entity khi spawn
        # Pose gốc để teleport (nếu có set_model_state)
        self.init_pose_xyzrpy = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # ------------ Service helpers ------------
        self.cli_pause, self.srv_pause = self._pick_empty(['/pause_physics', '/gazebo/pause_physics'])
        self.cli_unpause, self.srv_unpause = self._pick_empty(['/unpause_physics', '/gazebo/unpause_physics'])
        self.cli_reset_world, self.srv_reset_world = self._pick_empty(['/reset_world', '/gazebo/reset_world'])
        self.cli_set_state, self.srv_set_state = self._pick_set_model_state(['/set_model_state', '/gazebo/set_model_state'])
        self.back_eps = 0.05 * self.max_w      # ngưỡng coi là lùi (5% max_w)
        self.back_penalty_k = 0.6   

        self.get_logger().info(
            f'Using services: pause={self.srv_pause}, reset={self.srv_reset_world}, '
            f'unpause={self.srv_unpause}, set_state={self.srv_set_state}'
        )

    # -------------------- client pickers --------------------
    def _pick_empty(self, names):
        """Trả về (client, service_name) đầu tiên ready trong danh sách `names`."""
        for n in names:
            c = self.create_client(Empty, n)
            if c.wait_for_service(timeout_sec=0.5):
                return c, n
        # fallback: trả client đầu tiên (có thể chưa ready, sẽ chờ lúc gọi)
        return self.create_client(Empty, names[0]), names[0]

    def _pick_set_model_state(self, names):
        """Trả về (client, service_name) cho SetModelState nếu có; không thì (None, None)."""
        for n in names:
            c = self.create_client(SetModelState, n)
            if c.wait_for_service(timeout_sec=0.5):
                return c, n
        return None, None

    # -------------------- ROS Callbacks --------------------
    def image_cb(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv_image = cv2.resize(cv_image, (84, 84))
            self.latest_img = cv_image
            self.have_img = True
        except Exception as e:
            self.get_logger().warn(f'Image convert error: {e}')

    # -------------------- Gazebo Helpers --------------------
    def _call_empty(self, client, name: str, timeout: float = 2.0):
        if client is None:
            self.get_logger().warn(f'Client None for service {name}')
            return False
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().warn(f'Service unavailable: {name}')
            return False
        try:
            fut = client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
            return fut.result() is not None
        except Exception as e:
            self.get_logger().warn(f'Call failed {name}: {e}')
            return False

    def _teleport_to_start(self, timeout: float = 2.0):
        """Đưa model về pose gốc + xóa vận tốc bằng SetModelState (nếu có)."""
        if self.cli_set_state is None:
            return False
        if not self.cli_set_state.wait_for_service(timeout_sec=timeout):
            self.get_logger().warn(f'{self.srv_set_state} unavailable')
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

        state.twist = Twist()  # zero all velocities

        try:
            req = SetModelState.Request()
            req.model_state = state
            fut = self.cli_set_state.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
            res = fut.result()
            if res is None:
                self.get_logger().warn('set_model_state returned None')
                return False
            # Gazebo Classic có thể trả status_message
            if hasattr(res, 'status_message') and res.status_message:
                self.get_logger().info(f'set_model_state: {res.status_message}')
            return True
        except Exception as e:
            self.get_logger().warn(f'set_model_state failed: {e}')
            return False

    def _reset_world(self):
        """Pause → reset_world → (teleport nếu có set_model_state) → unpause."""
        # pause
        self._call_empty(self.cli_pause, self.srv_pause, timeout=2.0)

        # reset_world đưa models về spawn pose
        self._call_empty(self.cli_reset_world, self.srv_reset_world, timeout=2.0)

        # dừng bánh để tránh giật
        self.left_pub.publish(Float64MultiArray(data=[0.0]))
        self.right_pub.publish(Float64MultiArray(data=[0.0]))

        # teleport chỉ khi có service
        if self.cli_set_state is not None:
            ok = self._teleport_to_start(timeout=2.0)
            if not ok:
                self.get_logger().warn('Teleport failed; continuing after reset_world')
        else:
            self.get_logger().info('No set_model_state service; using reset_world pose.')

        # cho cảm biến ổn định
        for _ in range(3):
            rclpy.spin_once(self, timeout_sec=0.05)

        # unpause
        self._call_empty(self.cli_unpause, self.srv_unpause, timeout=2.0)

    # -------------------- Gym API --------------------
    def step(self, action):
        # Map action [-1,1] → rad/s
        wl_cmd = float(np.clip(action[0], -1.0, 1.0)) * self.max_w
        wr_cmd = float(np.clip(action[1], -1.0, 1.0)) * self.max_w

        # Publish velocity (Float64MultiArray – 1 phần tử)
        self.left_pub.publish(Float64MultiArray(data=[wl_cmd]))
        self.right_pub.publish(Float64MultiArray(data=[wr_cmd]))

        # Tiến thời gian step_dt
        t_end = time.time() + self.step_dt
        while rclpy.ok() and time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.0)

        obs = self.latest_img
        reward =0.0
        # ===== Reward: giữ trung tâm vạch vàng ở giữa ảnh =====
        hsv = cv2.cvtColor(obs, cv2.COLOR_BGR2HSV)
        # Khoảng màu vàng – chỉnh theo world cho chuẩn
        mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))

        cols = np.where(mask > 0)[1]
        center = obs.shape[1] // 2  # 42
        if cols.size > 0:
            lane_center = float(np.mean(cols))
            deviation = abs(lane_center - center) / center  # ~0..1
            reward += 1.0 - deviation
            off_lane = deviation > 0.95
        else:
            reward += -1.0
            off_lane = True

        if not np.isfinite(reward):
            reward += -1.0
            off_lane = True

        if (wl_cmd < -self.back_eps) and (wr_cmd < -self.back_eps):
            # Chuẩn hóa cường độ lùi về [0..1] theo max_w để công bằng
            back_strength = (abs(wl_cmd) + abs(wr_cmd)) / (2.0 * self.max_w)
            penalty_back = self.back_penalty_k * back_strength
            reward -= penalty_back
        omega_cmd = (wr_cmd - wl_cmd) * 0.05 / max(0.35, 1e-6)
        over = abs(omega_cmd) 
        if over > 0.6:
            reward -= 0.5

        self.step_count += 1
        terminated = bool(off_lane)
        truncated  = bool(self.step_count >= self.max_steps)
        info = {'wl': wl_cmd, 'wr': wr_cmd}

        # Gymnasium API: (obs, reward, terminated, truncated, info)
        return obs, float(reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Stop wheels trước khi reset
        self.left_pub.publish(Float64MultiArray(data=[0.0]))
        self.right_pub.publish(Float64MultiArray(data=[0.0]))

        # Reset world (+ teleport nếu có)
        self._reset_world()

        # Đợi khung hình đầu
        self.have_img = False
        t0 = time.time()
        while rclpy.ok() and not self.have_img and time.time() - t0 < 2.0:
            rclpy.spin_once(self, timeout_sec=0.1)

        # Gymnasium API: (obs, info)
        return self.latest_img, {}

    def close(self):
        try:
            self.left_pub.publish(Float64MultiArray(data=[0.0]))
            self.right_pub.publish(Float64MultiArray(data=[0.0]))
        except Exception:
            pass
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# -------------------- Test standalone --------------------
if __name__ == '__main__':
    env = LaneFollowerEnv(
        left_topic='/left_wheel_vel/commands',
        right_topic='/right_wheel_vel/commands',
        cam_topic='/camera/image_raw',
        max_wheel_rad_s=8.0,
        step_dt=0.05
    )

    try:
        obs, info = env.reset()
        print('Reset done. Obs shape:', obs.shape)
        # Run vài bước random
        for i in range(20):
            a = env.action_space.sample()
            obs, r, term, trunc, inf = env.step(a)
            print(f'step={i:02d} r={r:.3f} term={term} trunc={trunc} wl={inf["wl"]:.2f} wr={inf["wr"]:.2f}')
            if term or trunc:
                obs, info = env.reset()
    finally:
        env.close()
