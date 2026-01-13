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
import math

# --- 忽略 NumPy 的警告 ---
import warnings
warnings.filterwarnings("ignore")

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

        # --- 垂直待机姿态 ---
        # J2 = -1.57 (直指苍穹), J1设为0(默认)
        self.vertical_pose = [0.0, -1.57, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.create_timer(0.1, self.idle_callback)
        self.get_logger().info(" ⏳  等待 robot_description (URDF)...")

    def idle_callback(self):
        if not self.is_moving and self.current_joints is not None:
            self.publish_joints(self.current_joints, gripper_open=True)

    def urdf_callback(self, msg):
        if self.urdf_ready: return
        self.get_logger().info(" ✅  收到 URDF，正在解析...")
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.urdf') as tmp:
                tmp.write(msg.data)
                tmp_path = tmp.name

            temp_chain = ikpy.chain.Chain.from_urdf_file(tmp_path, base_elements=self.base_elements, name="openarm_left")
            self.mask = []
            for link in temp_chain.links:
                is_active = False
                if "joint" in link.name:
                    if "finger" not in link.name and "hand" not in link.name and "world" not in link.name and "link0" not in link.name:
                        is_active = True
                self.mask.append(is_active)

            self.ik_chain = ikpy.chain.Chain.from_urdf_file(
                tmp_path, base_elements=self.base_elements, name="openarm_left", active_links_mask=self.mask
            )
            self.urdf_ready = True
            os.remove(tmp_path)
            
            self.perform_reset()

        except Exception as e:
            self.get_logger().error(f"URDF 构建失败: {e}")

    def perform_reset(self):
        self.get_logger().info(" 💪  初始化：前往绝对安全高度 (竖直)...")
        self.is_moving = True
        self.move_smoothly(self.current_joints, self.vertical_pose, 2.5, True)
        self.is_moving = False
        self.get_logger().info(" 💪  就绪，等待香蕉坐标...")

    def solve_ik(self, x, y, z):
        target_pos = [x, y, z]
        
        # --- 强制垂直向下的旋转矩阵 ---
        # 绕X轴翻转180度，使Z轴向下
        target_orientation = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])

        try:
            # 垂直种子优化
            base_angle = math.atan2(y, x)
            perfect_seed = [0.0] * 7
            perfect_seed[0] = base_angle
            perfect_seed[1] = -0.5
            perfect_seed[3] = 1.5
            perfect_seed[5] = 0.5
            
            initial_full = [0.0] * len(self.ik_chain.links)
            j_idx = 0
            for i, is_active in enumerate(self.mask):
                if is_active:
                    if j_idx < 7: initial_full[i] = perfect_seed[j_idx]
                    j_idx += 1
            
            # 同时计算位置和姿态
            ik_solution = self.ik_chain.inverse_kinematics(
                target_position=target_pos, 
                target_orientation=target_orientation, 
                orientation_mode="all", 
                initial_position=initial_full
            )
            
            final_joints = []
            for i, angle in enumerate(ik_solution):
                if self.mask[i]: final_joints.append(angle)
            return final_joints
        except Exception:
            return None

    def target_callback(self, msg):
        if not self.urdf_ready or self.is_moving or self.mission_completed: return
        
        if 0.3 < msg.point.x < 0.7:
            self.get_logger().info(f" 🎯  锁定目标 X={msg.point.x:.2f}")
            self.execute_grasp(msg.point.x, msg.point.y, msg.point.z)

    def execute_grasp(self, x, y, z_floor):
        self.is_moving = True
        
        # --- 参数设定 ---
        safe_height = 0.80      
        grasp_height = z_floor + 0.05 
        lift_height = z_floor + 0.25   

        self.get_logger().info(" 🚀  开始【防碰撞优化版】抓取流程")

        # 1. 计算终点 (Pose Adjustment Target)
        # 这个 q_top 是在香蕉正上方，且夹爪垂直向下的姿态
        q_top = self.solve_ik(x, y, safe_height)
        
        if q_top:
            q_grasp = self.solve_ik(x, y, grasp_height)

            if q_grasp:
                # --- 构建两个关键中间姿态 ---
                
                # 姿态A: [原地] 垂直向上
                # 保持当前的 J1 (底座方向)，但把手臂竖起来
                q_lift_in_place = list(self.vertical_pose)
                if self.current_joints:
                    q_lift_in_place[0] = self.current_joints[0] # 保持底座不动
                
                # 姿态B: [对准] 垂直向上
                # 手臂依然竖直，但底座旋转到香蕉的方向 (Turret Mode)
                q_rotate_base = list(self.vertical_pose)
                q_rotate_base[0] = q_top[0] # 使用目标点的底座角度

                # --- 执行动作序列 ---

                # 1. 原地起飞：先把自己拔高，避免横扫
                self.get_logger().info("--> 1. 原地垂直起飞 (Safety Lift)")
                self.move_smoothly(self.current_joints, q_lift_in_place, 2.0, gripper_open=True)
                
                # 2. 炮塔旋转：在高空只转底座，对准目标
                self.get_logger().info("--> 2. 高空水平旋转 (Turret Turn)")
                self.move_smoothly(q_lift_in_place, q_rotate_base, 2.0, gripper_open=True)
                time.sleep(0.5)

                # 3. 姿态调整：在目标正上方，把夹爪翻下来 (Unfold)
                # 此时是从“竖直向上”过渡到“竖直向下”，因为高度够高(0.8m)，不会撞桌子
                self.get_logger().info("--> 3. 调整夹爪姿态 (Unfold Down)")
                self.move_smoothly(q_rotate_base, q_top, 2.0, gripper_open=True)
                time.sleep(0.5)

                # 4. 垂直下降抓取
                self.get_logger().info("--> 4. 垂直下降 (Descend)")
                self.move_smoothly(q_top, q_grasp, 2.0, gripper_open=True)
                time.sleep(0.5)

                # 5. 抓取 (闭合触发磁吸)
                self.get_logger().info("--> 5. 闭合夹爪 (Stick!)")
                self.publish_joints(q_grasp, gripper_open=False)
                time.sleep(1.5) 

                # 6. 提起
                self.get_logger().info("--> 6. 提起 (Lift)")
                q_lift = self.solve_ik(x, y, lift_height)
                if q_lift:
                    self.move_smoothly(q_grasp, q_lift, 1.5, gripper_open=False)
                    time.sleep(0.5)

                    # 7. 松开
                    self.get_logger().info("--> 7. 松开夹爪 (Drop)")
                    self.publish_joints(q_lift, gripper_open=True) # 0.04
                    time.sleep(1.0)
                    self.current_joints = list(q_lift)
                
                # 8. 结束
                self.get_logger().info(" ✨  任务完成")
                self.mission_completed = True 

            else:
                self.get_logger().error(" ⚠️   抓取点 IK 失败")
        else:
            self.get_logger().error(" ⚠️   高空点 IK 失败")
        
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
        self.publish_joints(end.tolist(), gripper_open)
        self.current_joints = list(end)

    def publish_joints(self, angles, gripper_open):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names + self.gripper_names
        
        # Open=0.04 (张开), Close=-0.02 (闭合触发磁吸)
        g_val = 0.04 if gripper_open else -0.02 
        
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
    rclpy.shutdown()

if __name__ == '__main__':
    main()
