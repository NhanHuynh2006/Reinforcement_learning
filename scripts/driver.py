import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

WHEEL_RADIUS = 0.05     # m  (khớp YAML)
WHEEL_SEP    = 0.35     # m  (khớp YAML)

def twist_to_wheels(v, w, L=WHEEL_SEP, r=WHEEL_RADIUS):
    # v: m/s, w: rad/s  ->  omega_L, omega_R (rad/s)
    v_L = v - w * (L / 2.0)
    v_R = v + w * (L / 2.0)
    return (v_L / r, v_R / r)

class WheelDriver(Node):
    def __init__(self,
                 left_topic='/left_wheel_vel/commands',
                 right_topic='/right_wheel_vel/commands',
                 rate_hz=20.0):
        super().__init__('wheel_driver')
        self.pub_L = self.create_publisher(Float64MultiArray, left_topic, 10)
        self.pub_R = self.create_publisher(Float64MultiArray, right_topic, 10)
        self.dt = 1.0 / rate_hz

    def send_wheels(self, wl, wr, duration_s):
        msgL = Float64MultiArray(data=[float(wl)])
        msgR = Float64MultiArray(data=[float(wr)])
        t_end = time.time() + duration_s
        while rclpy.ok() and time.time() < t_end:
            self.pub_L.publish(msgL)
            self.pub_R.publish(msgR)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(self.dt)
        # stop
        self.pub_L.publish(Float64MultiArray(data=[0.0]))
        self.pub_R.publish(Float64MultiArray(data=[0.0]))

    def send_twist(self, v, w, duration_s):
        wl, wr = twist_to_wheels(v, w)
        self.get_logger().info(f'Wheel rates (rad/s): L={wl:.2f}, R={wr:.2f}')
        self.send_wheels(wl, wr, duration_s)

def main():
    rclpy.init()
    node = WheelDriver(rate_hz=20.0)

    # Ví dụ 1: đặt trực tiếp tốc độ bánh (rad/s)
    node.get_logger().info('Both wheels 6.0 rad/s for 2s (đi thẳng ~0.3 m/s)')
    node.send_wheels(-6.0, -6.0, 5.0)



    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
