import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, JointState
import mujoco
import numpy as np
from cv_bridge import CvBridge
import time
import mujoco.viewer  # <--- 新增这行

class MujocoRosBridge(Node):
    def __init__(self):
        super().__init__('mujoco_ros_bridge')
        
        # --- 路径已修改为你自己的路径 ---
        xml_path = '/home/ferry/ros2_course_work/src/openarm_description/mujoco/scene_with_table.xml'
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        # --- 新增：启动原生可视化窗口 ---
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        self.joint_positions = {}
        
        self.img_pub = self.create_publisher(Image, '/depth_camera/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/depth_camera/camera_info', 10)
        self.bridge = CvBridge()
        
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        
        self.camera_info = self.get_camera_info(640, 480, 50.0) # 注意：这里FOV要和你XML里设的一样，假设是50
        self.timer = self.create_timer(0.01, self.timer_callback)

    def get_camera_info(self, width, height, fovy):
        info = CameraInfo()
        f = height / (2 * np.tan(np.deg2rad(fovy) / 2))
        info.width = width
        info.height = height
        info.distortion_model = "plumb_bob"
        info.k = [f, 0.0, width/2, 0.0, f, height/2, 0.0, 0.0, 1.0]
        info.p = [f, 0.0, width/2, 0.0, 0.0, f, height/2, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def joint_callback(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions[name] = pos

    def timer_callback(self):
        # 同步关节角度
        for name, pos in self.joint_positions.items():
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id != -1:
                    qpos_adr = self.model.jnt_qposadr[joint_id]
                    self.data.qpos[qpos_adr] = pos
            except:
                continue

        mujoco.mj_forward(self.model, self.data)
        # --- 新增：刷新窗口 ---
        self.viewer.sync()
        
        # 渲染并发布
        self.renderer.update_scene(self.data, camera="depth_camera")
        rgb_image = self.renderer.render()
        img_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='rgb8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'depth_camera'
        
        self.img_pub.publish(img_msg)
        self.camera_info.header = img_msg.header
        self.info_pub.publish(self.camera_info)

def main():
    rclpy.init()
    node = MujocoRosBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
