# Train RL with Segmentation Input

Mục tiêu: xe chạy đúng làn **bên phải** (giữa vạch vàng và vạch trắng bên phải). Nếu ra làn hoặc chạm vật cản thì bị trừ điểm / reset.

## 1) Chạy mô phỏng
Terminal 1:

```
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=0
export GAZEBO_IP=127.0.0.1
export GAZEBO_MASTER_URI=http://127.0.0.1:11345

source /opt/ros/humble/setup.bash
source ~/Documents/Reinforcement_Learning/install/setup.bash

__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
ros2 launch Reinforcement_Learning launch_sim_tg3_train.launch.py
```

## 2) Cài Python deps (nếu thiếu)
```
pip install stable-baselines3 gymnasium opencv-python ultralytics
```

> `ultralytics` dùng cho YOLOv8-seg. Nếu chưa có weights, bạn có thể để trống để dùng mask màu (fallback).

## 3) Train
Terminal 2 (cùng môi trường ROS):

```
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=0
export GAZEBO_IP=127.0.0.1
export GAZEBO_MASTER_URI=http://127.0.0.1:11345

source /opt/ros/humble/setup.bash
source ~/Documents/Reinforcement_Learning/install/setup.bash

source train/bin/activate

python3 train/train_ppo_seg.py \
--logdir ~/runs/lane_rl_seg_new \
--publish-debug \
--obs-size 160 \
--yolo-weights /home/nolan/Documents/Yolo_v11/runs/segment/runs/yolo11_seg_best_stable/weights/best.pt \
--yolo-device 0 \
--yolo-every 2 \
--yolo-imgsz 432
```

Nếu chưa có YOLO weights, bỏ `--yolo-weights` để dùng fallback.

## 4) Evaluate
```
python3 train/eval_ppo_seg.py --model ~/runs/lane_rl_seg/ppo_lane_seg_final.zip
```

## Gợi ý chỉnh tham số
- `--max-v`: tốc độ thẳng tối đa
- `--max-w`: tốc độ quay tối đa
- `--step-dt`: thời gian mỗi step (giây)
- `--obs-size`: kích thước ảnh mask (mặc định 84)

## Lưu ý
- Nếu `spawn_entity` hoặc `controller_manager` lỗi, hãy chạy lại launch và kiểm tra `/spawn_entity` service.
- Train lâu: bắt đầu 200k steps, sau đó tăng 1-2 triệu.
## 5) Debug
```
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=0
export GAZEBO_IP=127.0.0.1
export GAZEBO_MASTER_URI=http://127.0.0.1:11345

source /opt/ros/humble/setup.bash
source ~/Documents/Reinforcement_Learning/install/setup.bash

ros2 run rqt_image_view rqt_image_view
```