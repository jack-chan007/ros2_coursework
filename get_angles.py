import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class GetCurrentAngles(Node):
    def __init__(self):
        super().__init__('get_current_angles')
        # 订阅关节状态话题
        self.sub = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.listener_callback, 
            10
        )
        # 定义你需要关注的关节名称（顺序必须和 grasp_controller.py 里一致）
        self.target_joints = [
            'openarm_left_joint1', 
            'openarm_left_joint2', 
            'openarm_left_joint3',
            'openarm_left_joint4', 
            'openarm_left_joint5', 
            'openarm_left_joint6',
            'openarm_left_joint7'
        ]
        self.get_logger().info("正在等待关节数据...请确保仿真正在运行...")

    def listener_callback(self, msg):
        current_angles = []
        found_all = True
        
        # 从消息中查找对应的关节角度
        for target_name in self.target_joints:
            if target_name in msg.name:
                idx = msg.name.index(target_name)
                # 保留4位小数，精度足够了
                angle = round(msg.position[idx], 4)
                current_angles.append(angle)
            else:
                found_all = False
                break
        
        if found_all:
            print("\n" + "="*40)
            print("✅ 成功获取当前姿态！请复制下面的列表：")
            print("-" * 40)
            print(f"self.grasp_angles = {current_angles}")
            print("-" * 40)
            print("="*40 + "\n")
            
            #以此为目的，获取一次后直接退出
            raise SystemExit

def main(args=None):
    rclpy.init(args=args)
    node = GetCurrentAngles()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
