import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from rclpy.qos import QoSProfile, DurabilityPolicy

import tempfile
import os
import ikpy.chain
import numpy as np
import time

class AutoGraspController(Node):
    def __init__(self):
        super().__init__('auto_grasp_controller')

        # 1. 基础配置
        qos_profile = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description', self.urdf_callback, qos_profile)
        self.create_subscription(PointStamped, '/banana_position', self.target_callback, 10)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        self.base_elements = ["world"]
        self.joint_names = [
            'openarm_left_joint1', 'openarm_left_joint2', 'openarm_left_joint3',
            'openarm_left_joint4', 'openarm_left_joint5', 'openarm_left_joint6',
            'openarm_left_joint7'
        ]
        self.gripper_names = ['openarm_left_finger_joint1', 'openarm_left_finger_joint2']

        self.ik_chain = None
        self.mask = []
        self.current_joints = [0.0] * 7 
        self.is_moving = False
        self.urdf_ready = False
        self.mission_completed = False

        # --- 【关键定义】绝对安全的固定姿态 ---
        
        # 1. 左侧放置点 (高空，防止放下时撞桌子)
        # J1=1.57(左), J2=-0.8(抬高), J4=0.5(微弯)
        self.place_pose = [1.57, -0.8, 0.0, 0.5, 0.0, 0.0, 0.0]
        
        # 2. 中间高空过渡点 (复位前的安全点)
        # J1=0.0(回正), J2=-1.0(举高), J4=0.0(伸直)
        self.high_center_pose = [0.0, -1.57, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # 3. 最终待机姿态 (自然下垂)
        self.reset_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.create_timer(0.1, self.idle_callback)
        self.get_logger().info("⏳ 等待 robot_description...")

    def idle_callback(self):
        if not self.is_moving and self.current_joints is not None:
            self.publish_joints(self.current_joints, gripper_open=True)

    def urdf_callback(self, msg):
        if self.urdf_ready: return
        self.get_logger().info("✅ 收到 URDF...")
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.urdf') as tmp:
                tmp.write(msg.data)
                tmp_path = tmp.name
            
            # Mask
            temp_chain = ikpy.chain.Chain.from_urdf_file(tmp_path, base_elements=self.base_elements, name="openarm_left")
            self.mask = []
            for link in temp_chain.links:
                is_active = False
                if "joint" in link.name:
                    if "finger" not in link.name and "hand" not in link.name and "world" not in link.name and "link0" not in link.name:
                         is_active = True
                self.mask.append(is_active)
            
            # Chain
            self.ik_chain = ikpy.chain.Chain.from_urdf_file(
                tmp_path, base_elements=self.base_elements, name="openarm_left", active_links_mask=self.mask
            )
            self.urdf_ready = True
            os.remove(tmp_path)
            
            # 自检
            self.perform_wiggle_test()
            
        except Exception as e:
            self.get_logger().error(f"构建失败: {e}")

    def perform_wiggle_test(self):
        self.get_logger().info("💪 自检：前往高空安全点...")
        self.is_moving = True
        self.move_smoothly(self.current_joints, self.high_center_pose, 2.0, True)
        self.is_moving = False
        self.get_logger().info("💪 就绪")

    def solve_ik(self, x, y, z):
        target_pos = [x, y, z]
        try:
            initial_full = [0.0] * len(self.ik_chain.links)
            j_idx = 0
            for i, is_active in enumerate(self.mask):
                if is_active and j_idx < len(self.current_joints):
                    initial_full[i] = self.current_joints[j_idx]
                    j_idx += 1
            
            ik_solution = self.ik_chain.inverse_kinematics(target_position=target_pos, initial_position=initial_full)
            final_joints = []
            for i, angle in enumerate(ik_solution):
                if self.mask[i]: final_joints.append(angle)
            return final_joints
        except Exception:
            return None

    def target_callback(self, msg):
        if not self.urdf_ready or self.is_moving or self.mission_completed: return
        if 0.3 < msg.point.x < 0.7:
            self.get_logger().info(f"🎯 锁定 X={msg.point.x:.2f}")
            self.execute_grasp(msg.point.x, msg.point.y, msg.point.z)

    def execute_grasp(self, x, y, z_floor):
        # --- 安全检查 ---
        if z_floor < 0.3:
            self.get_logger().error(f"❌ 目标高度异常 (Z={z_floor:.2f})，疑似检测错误，取消抓取！")
            return

        self.is_moving = True
        self.get_logger().info(f"1. 计算抓取点 (基准Z={z_floor:.2f})...")
        
        # 保持之前的 Z 轴微调
        grasp_height = z_floor + 0.07 
        pre_grasp_height = z_floor + 0.20

        q_grasp = self.solve_ik(x, y, grasp_height)
        q_pre   = self.solve_ik(x, y, pre_grasp_height)

        if q_grasp and q_pre:
            self.get_logger().info(" 🚀  执行【超高空避障】抓取序列")

            # --- 1. 前往【超高】安全点 ---
            # 这一步会把手臂完全竖直，确保彻底避开桌子
            self.get_logger().info("--> 1. 前往高空安全点")
            self.move_smoothly(self.current_joints, self.high_center_pose, 2.0, True)
            time.sleep(0.2)

            # --- 2. 移动到物体正上方 ---
            self.get_logger().info("--> 2. 移动到物体正上方")
            self.move_smoothly(self.high_center_pose, q_pre, 2.0, True)
            time.sleep(0.5)

            # --- 3. 垂直下降 ---
            self.get_logger().info("--> 3. 垂直下降")
            self.move_smoothly(q_pre, q_grasp, 1.5, True)
            time.sleep(0.5)

            # --- 4. 闭合夹爪 ---
            self.get_logger().info("--> 4. 闭合夹爪")
            self.publish_joints(q_grasp, gripper_open=False)
            time.sleep(1.0)

            # --- 5. 垂直起飞 (修改点：飞得更高) ---
            q_stiff_lift = list(q_grasp)
            # 这里改为 -1.57，抓完直接举火炬一样举起来
            q_stiff_lift[1] = -1.57  
            q_stiff_lift[3] = 0.0    # 手臂伸直

            self.get_logger().info("--> 5. 垂直起飞 (最高点)")
            self.move_smoothly(q_grasp, q_stiff_lift, 2.0, False)
            time.sleep(0.5)

            # --- 6. 搬运至左侧 ---
            self.get_logger().info("--> 6. 搬运至左侧")
            self.move_smoothly(q_stiff_lift, self.place_pose, 2.5, False)
            time.sleep(0.5)

            # --- 7. 放下物体 ---
            self.get_logger().info("--> 7. 放下物体")
            self.publish_joints(self.place_pose, gripper_open=True)
            time.sleep(1.0)

            # --- 8. 复位 ---
            self.get_logger().info("--> 8. 空中回正与复位")
            # 先回到最高点，再复位，防止扫倒其他东西
            self.move_smoothly(self.place_pose, self.high_center_pose, 2.0, True)
            self.move_smoothly(self.high_center_pose, self.reset_pose, 2.0, True)

            self.get_logger().info(" ✨  任务完成!")
            self.mission_completed = True

        else:
            self.get_logger().error(" ⚠️  IK 计算失败")
            self.is_moving = False
            
    def move_smoothly(self, start, end, duration, gripper_open):
        steps = int(duration * 50)
        start = np.array(start)
        end = np.array(end)
        for i in range(steps):
            progress = (i + 1) / steps
            interp = start + (end - start) * progress
            self.publish_joints(interp.tolist(), gripper_open)
            time.sleep(duration / steps)
        self.current_joints = list(end)

    def publish_joints(self, angles, gripper_open):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names + self.gripper_names
        # 强力夹紧
        g_val = 0.0 if gripper_open else -0.02
        msg.position = angles + [g_val, g_val]
        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AutoGraspController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()

if __name__ == '__main__':
    main()
