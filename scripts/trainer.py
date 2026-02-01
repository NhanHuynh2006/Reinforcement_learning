#!/usr/bin/env python3
import os
import time
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# import env của bạn (đoạn đã gửi “điều khiển 2 bánh”)
from env_rl import LaneFollowerEnv  # đổi tên theo file bạn lưu

LOGDIR = os.path.expanduser("~/runs/lane_rl_1")
os.makedirs(LOGDIR, exist_ok=True)

def make_env():
    # mỗi env là một ROS node → phải init rclpy trước
    def _thunk():
        if not rclpy.ok():
            rclpy.init(args=None)
        # cam_topic có thể đổi ở đây nếu cần
        env = LaneFollowerEnv(
            left_topic='/left_wheel_vel/commands',
            right_topic='/right_wheel_vel/commands',
            cam_topic='/camera/image_raw',
            max_wheel_rad_s=8.0,
            step_dt=0.05
        )
        return Monitor(env, filename=os.path.join(LOGDIR, "monitor.csv"))
    return _thunk

def main():
    # 1 env (ROS không hợp multi-process)
    env = DummyVecEnv([make_env()])

    # Model PPO với policy CNN cho ảnh
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,           # ~5.12s/rollout với step_dt=0.05
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        clip_range=0.2,
        tensorboard_log=LOGDIR,
        verbose=1,
    )

    # Callback: lưu checkpoint mỗi 50k steps
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // env.num_envs,
        save_path=LOGDIR,
        name_prefix="ppo_lane"
    )

    # (tùy chọn) Eval callback: dùng cùng env cho đơn giản
    eval_cb = EvalCallback(
        env,
        best_model_save_path=LOGDIR,
        log_path=LOGDIR,
        eval_freq=25_000 // env.num_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    total_steps = 20_000  # tăng dần lên 1–2 triệu khi ổn định
    model.learn(total_timesteps=total_steps, callback=[checkpoint_cb, eval_cb], progress_bar=True)

    model.save(os.path.join(LOGDIR, "ppo_lane_final"))
    print(f"Saved to {LOGDIR}/ppo_lane_final.zip")

    # tắt ROS node gọn gàng
    try:
        env.envs[0].close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
