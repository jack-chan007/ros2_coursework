import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy

import PyKDL
from kdl_parser_py.urdf import treeFromString
import time
import numpy as np

class KDLGraspController(Node):
    def __init__(self):
        super().__init__('kdl_grasp_controller')

        # 1. 订阅 URDF (为了建立运动学模型)
        qos_profile = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description', self.urdf_callback, qos_profile)

        # 2. 订阅视觉目标
        self.create_subscription(PointStamped, '/banana_position', self.target_callback, 10)

        # 3. 订阅当前关节状态 (IK求解需要当前角度作为初值)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # 4. 发布控制指令
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        # --- 配置区 (根据你的模型修改) ---
        self.base_link = "world"                  # 运动链起点
        self.tip_link = "openarm_left_hand_tcp"   # 运动链终点 (抓手中心)
        
        self.joint_names = [
            'openarm_left_joint1', 'openarm_left_joint2', 'openarm_left_joint3',
            'openarm_left_joint4', 'openarm_left_joint5', 'openarm_left_joint6',
            'openarm_left_joint7'
        ]
        self.gripper_names = ['openarm_left_finger_joint1', 'openarm_left_finger_joint2']
        
        # 状态变量
        self.kdl_chain = None
        self.ik_solver = None
        self.fk_solver = None
        self.current_joints = None
        self.urdf_loaded = False
        self.is_moving = False
        
        self.get_logger().info("⏳ 等待 robot_description (URDF)...")

    def urdf_callback(self, msg):
        if self.urdf_loaded: return
        
        self.get_logger().info("✅ 收到 URDF! 正在构建 KDL 运动链...")
        success, tree = treeFromString(msg.data)
        if not success:
            self.get_logger().error("无法从 URDF 解析 KDL 树")
            return

        # 提取从 world 到 tcp 的链条
        self.kdl_chain = tree.getChain(self.base_link, self.tip_link)
        
        # 创建求解器
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(self.kdl_chain)
        self.ik_vel_solver = PyKDL.ChainIkSolverVel_pinv(self.kdl_chain)
        # 牛顿-拉夫逊迭代法求解 IK
        self.ik_solver = PyKDL.ChainIkSolverPos_NR(self.kdl_chain, self.fk_solver, self.ik_vel_solver)
        
        self.urdf_loaded = True
        self.get_logger().info(f"✅ KDL 初始化完成! 关节数: {self.kdl_chain.getNrOfJoints()}")

    def joint_state_callback(self, msg):
        # 实时更新当前关节角度，用于 IK 的种子值
        tmp_vals = []
        for name in self.joint_names:
            if name in msg.name:
                idx = msg.name.index(name)
                tmp_vals.append(msg.position[idx])
        
        if len(tmp_vals) == 7:
            self.current_joints = tmp_vals

    def target_callback(self, msg):
        if not self.urdf_loaded or self.is_moving or self.current_joints is None:
            return

        # 简单的状态机：发现香蕉 -> 求解 IK -> 执行
        # 过滤一下范围，防止误触发
        if 0.3 < msg.point.x < 0.7:
            self.get_logger().info(f"🎯 视觉锁定香蕉: ({msg.point.x:.2f}, {msg.point.y:.2f}, {msg.point.z:.2f})")
            self.execute_automatic_grasp(msg.point.x, msg.point.y, msg.point.z)

    def solve_ik(self, x, y, z):
        # 1. 定义目标位置 (Vector)
        target_pos = PyKDL.Vector(x, y, z)
        
        # 2. 定义目标姿态 (Rotation)
        # 这里最关键！我们要让夹爪“垂直向下”抓取
        # 你可能需要根据实际情况调整这个旋转矩阵
        # 这里的 M 是一个让 Z 轴朝下的旋转矩阵示例
        target_rot = PyKDL.Rotation.RPY(0, 3.14159, 0) # 翻转180度向下
        
        target_frame = PyKDL.Frame(target_rot, target_pos)

        # 3. 准备初值 (种子)
        initial_q = PyKDL.JntArray(7)
        for i, val in enumerate(self.current_joints):
            initial_q[i] = val

        # 4. 求解
        result_q = PyKDL.JntArray(7)
        ret = self.ik_solver.CartToJnt(initial_q, target_frame, result_q)

        if ret >= 0:
            return [result_q[i] for i in range(7)] # 成功，返回角度列表
        else:
            self.get_logger().warn("⚠️ IK 求解失败! 目标可能不可达")
            return None

    def execute_automatic_grasp(self, x, y, z_floor):
        self.is_moving = True
        
        # 策略：预抓取点(上方) -> 抓取点(物体处) -> 闭合 -> 抬起
        
        # A. 计算 预抓取点 (香蕉上方 15cm)
        self.get_logger().info("1. 计算预抓取点 IK...")
        q_pre = self.solve_ik(x, y, z_floor + 0.15)
        
        # B. 计算 抓取点 (稍微抬高一点点防止撞桌子，比如 +0.02)
        self.get_logger().info("2. 计算抓取点 IK...")
        # 注意：这里可能需要微调 Z 值，这取决于你的 TCP 坐标系是在指尖还是手掌中心
        # 如果 TCP 在手掌，这里可能需要减去指尖长度；如果 TCP 在指尖，直接用物体高度
        q_grasp = self.solve_ik(x, y, z_floor + 0.05) 

        if q_pre and q_grasp:
            # 执行序列
            self.get_logger().info("🚀 开始自动抓取序列!")
            
            # 1. 去上方
            self.move_smoothly(self.current_joints, q_pre, 2.0, True)
            time.sleep(0.5)
            
            # 2. 下去
            self.move_smoothly(q_pre, q_grasp, 1.5, True)
            time.sleep(0.5)
            
            # 3. 闭合
            self.get_logger().info("👌 闭合夹爪")
            self.publish_joints(q_grasp, gripper_open=False)
            time.sleep(1.0)
            
            # 4. 抬起 (回到预抓取点)
            self.move_smoothly(q_grasp, q_pre, 1.5, False)
            self.get_logger().info("✨ 任务完成!")
            
        else:
            self.get_logger().error("IK 无解，放弃本次抓取")

        self.is_moving = False

    def move_smoothly(self, start_angles, end_angles, duration, gripper_open):
        # 简单的线性插值平滑控制
        steps = int(duration * 50)
        dt = duration / steps
        start = np.array(start_angles)
        end = np.array(end_angles)

        for i in range(steps):
            progress = (i + 1) / steps
            interp_angles = start + (end - start) * progress
            self.publish_joints(interp_angles.tolist(), gripper_open)
            time.sleep(dt)
        self.current_joints = list(end_angles)

    def publish_joints(self, arm_angles, gripper_open=True):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names + self.gripper_names
        gripper_val = 0.0 if gripper_open else 0.02
        msg.position = arm_angles + [gripper_val, gripper_val]
        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = KDLGraspController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
