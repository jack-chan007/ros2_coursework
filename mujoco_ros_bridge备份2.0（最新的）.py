import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, JointState
import mujoco
import mujoco.viewer
import numpy as np
from cv_bridge import CvBridge
import os

class MujocoRosBridge(Node):
    def __init__(self):
        super().__init__('mujoco_ros_bridge')
        
        # 1. 加载模型 (注意路径是否正确)
        xml_path = '/home/ferry/ros2_course_work/src/openarm_description/mujoco/scene_with_table.xml'
        
        # 检查文件是否存在，避免报错
        if not os.path.exists(xml_path):
             self.get_logger().error(f"❌ XML 文件未找到: {xml_path}")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        # 启动可视化窗口
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.joint_positions = {}
        
        self.img_pub = self.create_publisher(Image, '/depth_camera/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/depth_camera/camera_info', 10)
        self.bridge = CvBridge()
        
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        
        self.camera_info = self.get_camera_info(640, 480, 60.0)
        self.timer = self.create_timer(0.01, self.timer_callback)

        # 缓存 ID，用于磁吸逻辑
        try:
            self.banana_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "banana")
            # 注意：如果你的XML里夹爪末端名字不同，这里可能需要调整，比如 "openarm_left_link7"
            self.gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_left_hand_tcp") 
            if self.gripper_id == -1:
                self.gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_left_link7")
            
            # 获取香蕉自由关节的地址 (用于修改位置)
            banana_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "banana")
            self.banana_qpos_adr = self.model.jnt_qposadr[banana_joint_id]
            self.banana_qvel_adr = self.model.jnt_dofadr[banana_joint_id] # 速度地址
        except:
            self.get_logger().warn("⚠️ 无法找到香蕉或夹爪的ID，磁吸逻辑可能失效。")
            self.banana_id = -1

    def get_camera_info(self, width, height, fovy):
        info = CameraInfo()
        f = height / (2 * np.tan(np.deg2rad(fovy) / 2))
        info.width, info.height = width, height
        info.distortion_model = "plumb_bob"
        info.k = [f, 0.0, width/2, 0.0, f, height/2, 0.0, 0.0, 1.0]
        info.p = [f, 0.0, width/2, 0.0, 0.0, f, height/2, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def joint_callback(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions[name] = pos

    def timer_callback(self):
        # 1. 【同步机械臂关节】
        # 即使使用了 mj_step，我们依然在这里强行覆盖关节角度，
        # 这样机械臂就会乖乖听 ROS 的话，而不会因为重力软掉。
        gripper_closed = False
        for name, pos in self.joint_positions.items():
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id != -1:
                    qpos_adr = self.model.jnt_qposadr[joint_id]
                    self.data.qpos[qpos_adr] = pos
                
                # 检测夹爪是否闭合 (根据 openarm 的定义，负值通常是闭合)
                if "finger_joint1" in name and pos < 0.02:#原本是-0.01,现在改成0.02
                    gripper_closed = True
            except:
                continue

        # 2. 【磁吸逻辑 (The Magnet)】
        if self.banana_id != -1 and self.gripper_id != -1:
            # 获取位置
            banana_pos = self.data.xpos[self.banana_id]
            gripper_pos = self.data.xpos[self.gripper_id]
            
            # 计算距离
            dist = np.linalg.norm(banana_pos - gripper_pos)
            
            # 触发条件：夹爪闭合 且 距离足够近 (< 0.15m)
            if gripper_closed and dist < 0.15:
                # 强行设置香蕉的位置到夹爪下方
                # 这里的 [0.0, 0.0, 0.02] 是相对于夹爪中心的偏移量，需要根据模型微调
                # 如果香蕉穿模太严重，把 0.02 改大一点
                target_pos = gripper_pos + np.array([0.0, 0.0, 0.02]) 
                
                # 修改香蕉的位置 (qpos)
                self.data.qpos[self.banana_qpos_adr] = target_pos[0]
                self.data.qpos[self.banana_qpos_adr+1] = target_pos[1]
                self.data.qpos[self.banana_qpos_adr+2] = target_pos[2]
                
                # 【关键】将香蕉的速度 (qvel) 归零，防止它在被抓住时积累巨大的动量
                # 这样松开时它会垂直下落，而不是飞出去
                if hasattr(self, 'banana_qvel_adr'):
                    for i in range(6): # 自由关节有6个自由度
                        self.data.qvel[self.banana_qvel_adr + i] = 0.0

        # 3. 【核心修改】物理步进
        # 使用 mj_step 替代 mj_forward。
        # mj_step 会推进时间，让重力生效。这样当你松开夹爪（上面的if不执行）时，
        # 物理引擎的重力就会接管香蕉，让它掉下去。
        mujoco.mj_step(self.model, self.data)
        
        # 4. 更新画面
        self.viewer.sync()
        
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
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
