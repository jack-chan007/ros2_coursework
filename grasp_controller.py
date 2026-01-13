import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
import time

class GraspController(Node):
    def __init__(self):
        super().__init__('grasp_controller')

        # 1. 订阅视觉识别结果 (任务2发布的那个Topic)
        self.target_sub = self.create_subscription(
            PointStamped, '/banana_position', self.target_callback, 10)

        # 2. 发布关节角度控制机械臂 (直接发给桥接节点)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        # --- 关键配置 (请根据你的实际测量修改这里！) ---
        # 关节名称列表 (注意顺序要和滑块界面一致)
        self.joint_names = [
            'openarm_left_joint1', 'openarm_left_joint2', 'openarm_left_joint3',
            'openarm_left_joint4', 'openarm_left_joint5', 'openarm_left_joint6',
            'openarm_left_joint7'
        ]
        
        # 夹爪关节名称 (假设是这两个，如果你的模型不一样请修改)
        # 夹爪关节名称 (根据你的 topic echo 结果修改)
        self.gripper_names = ['openarm_left_finger_joint1', 'openarm_left_finger_joint2']

        # [姿态1]：初始等待位 (比如高高举起)
        self.home_angles = [0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0]

        # [姿态2]：抓取位 (这是你要填的最重要的部分！！！)
        # 把你在 GUI 上试出来的、能抓到香蕉的那一组角度填在这里
        self.grasp_angles = [-1.1132, 0.0614, -0.0254, 0.0, 0.0, -0.0127, 0.1954]

        # 状态机标志位
        self.is_moving = False
        self.has_grasped = False

        # 初始化机械臂到 Home 位置
        self.publish_joints(self.home_angles, gripper_open=True)
        self.get_logger().info("🤖 抓取控制器就绪，等待视觉信号...")

    def target_callback(self, msg):
        if self.is_moving or self.has_grasped:
            return

        # 简单的逻辑：如果视觉检测到了，且位置在合理范围内，就开始抓
        # msg.point.x 就是你之前算出来的 0.493
        if 0.45 < msg.point.x < 0.55:
            self.get_logger().info(f"👀 发现目标在 X={msg.point.x:.2f}，开始执行抓取序列！")
            self.execute_grasp_sequence()

    def execute_grasp_sequence(self):
        self.is_moving = True

        # 步骤 1: 移动到抓取位置 (Grasp Pose)
        self.get_logger().info("--> 1. 机械臂下放...")
        self.publish_joints(self.grasp_angles, gripper_open=True)
        time.sleep(3.0) # 等待运动到位

        # 步骤 2: 闭合夹爪 (Gripper Close)
        self.get_logger().info("--> 2. 闭合夹爪...")
        self.publish_joints(self.grasp_angles, gripper_open=False)
        time.sleep(1.0) # 等待夹紧

        # 步骤 3: 抬起机械臂 (Home Pose)
        self.get_logger().info("--> 3. 抬起物体...")
        self.publish_joints(self.home_angles, gripper_open=False) # 保持夹紧状态抬起
        time.sleep(2.0)

        self.get_logger().info("✅ 抓取完成！")
        self.has_grasped = True
        self.is_moving = False

    def publish_joints(self, arm_angles, gripper_open=True):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # 合并机械臂关节和夹爪关节
        msg.name = self.joint_names + self.gripper_names
        
        # 夹爪角度：假设 0.0 是开，0.04 是关 (具体数值可能需要微调)
        gripper_val = 0.0 if gripper_open else 0.04 
        gripper_pos = [gripper_val, gripper_val] # 两个手指

        msg.position = arm_angles + gripper_pos
        
        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = GraspController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
