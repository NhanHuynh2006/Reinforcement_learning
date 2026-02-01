#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer_realtime')
        # ⚠️ Thay topic này bằng topic camera thực tế của bạn
        self.topic_name = '/camera/image_raw'
        self.sub = self.create_subscription(Image, self.topic_name, self.image_callback, 10)
        self.bridge = CvBridge()
        self.last_time = time.time()
        self.frame_count = 0
        self.fps_display_interval = 1.0  # giây
        self.fps = 0
        self.get_logger().info(f"Subscribed to {self.topic_name}")

    def image_callback(self, msg):
        # Chuyển ROS Image -> OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Tính FPS trung bình mỗi giây
        self.frame_count += 1
        now = time.time()
        if now - self.last_time > self.fps_display_interval:
            self.fps = self.frame_count / (now - self.last_time)
            self.last_time = now
            self.frame_count = 0

        # Ghi FPS lên khung hình
        cv2.putText(cv_image, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Hiển thị cửa sổ realtime
        cv2.imshow("Gazebo Camera (Realtime)", cv_image)
        # 1ms chờ để cập nhật cửa sổ, ESC để thoát
        key = cv2.waitKey(1)
        if key == 27:
            self.get_logger().info("ESC pressed, closing viewer...")
            rclpy.shutdown()

def main():
    rclpy.init()
    node = CameraViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
