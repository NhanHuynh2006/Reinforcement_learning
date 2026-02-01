#!/usr/bin/env python3
import time
import rclpy
import numpy as np
from stable_baselines3 import PPO
from env_rl import LaneFollowerEnv 

def main():
    rclpy.init()
    env = LaneFollowerEnv(cam_topic='/camera/image_raw', step_dt=0.05)
    model = PPO.load("/home/min_tan/runs/lane_rl_1/ppo_lane_final.zip")

    obs, _ = env.reset()
    try:
        while rclpy.ok():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                obs, _ = env.reset()
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    main()
