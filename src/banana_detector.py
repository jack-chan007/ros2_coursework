import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import Buffer, TransformListener

class BananaDetector(Node):
    def __init__(self):
        super().__init__('banana_detector')

        # 1. 订阅图像和相机信息
        self.img_sub = self.create_subscription(
            Image, '/depth_camera/image_raw', self.image_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/depth_camera/camera_info', self.info_callback, 10)
        
        # 2. 发布物体位置 (给抓取节点用) 和 可视化Marker (给RViz看)
        self.target_pub = self.create_publisher(PointStamped, '/banana_position', 10)
        self.marker_pub = self.create_publisher(Marker, '/banana_marker', 10)

        self.bridge = CvBridge()
        self.camera_model = None
        
        # 3. 初始化 TF 监听器 (用于坐标变换)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 4. 桌面高度假设 (根据XML文件: table pos z=0.4 + size z=0.02 = 0.42m)
        self.table_height = 0.425  # 稍微加一点点，对应香蕉中心高度

        self.get_logger().info("🍌 香蕉检测节点已启动！等待图像...")

    def info_callback(self, msg):
        # 获取相机内参只需要一次
        if self.camera_model is None:
            self.camera_model = msg
            self.get_logger().info("收到相机内参！")

    def image_callback(self, msg):
        if self.camera_model is None:
            return

        # --- A. 图像处理 ---
        try:
            # ROS Image -> OpenCV Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")
            return

        # 转换为 HSV 进行颜色识别
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # 设定黄色的阈值 (根据实际光照可能需要微调)
        # OpenCV中 H: 0-179, S: 0-255, V: 0-255
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 腐蚀和膨胀去除噪点
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # 找到最大的轮廓 (假设是香蕉)
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            
            if M["m00"] > 0:
                # 计算像素中心 (u, v)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # 在图像上画个圈和点 (可选，用于调试)
                # cv2.circle(cv_image, (cX, cY), 5, (0, 0, 255), -1)
                # cv2.imshow("Detection", cv_image)
                # cv2.waitKey(1)

                # --- B. 坐标计算 (单目测距) ---
                self.process_coordinates(cX, cY, msg.header)

    def process_coordinates(self, u, v, header):
        # 1. 获取内参 fx, fy, cx, cy
        # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        fx = self.camera_model.k[0]
        fy = self.camera_model.k[4]
        cx = self.camera_model.k[2]
        cy = self.camera_model.k[5]

        # 2. 计算归一化坐标 (Z=1时的相机坐标)
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy
        
        # 创建一个 PointStamped，表示相机坐标系下的一个向量
        # 仅仅是方向向量，我们暂时不知道深度 Z
        camera_point = PointStamped()
        camera_point.header = header
        camera_point.point.x = x_norm
        camera_point.point.y = y_norm
        camera_point.point.z = 1.0  # 假设单位深度

        try:
            # 3. 查询 TF 变换: depth_camera -> world
            # 注意：我们要找的是此时此刻的变换，timeout设为1秒
            transform = self.tf_buffer.lookup_transform(
                'world', 
                header.frame_id, 
                rclpy.time.Time(), 
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            # 4. 将向量转换到 world 坐标系
            # 这一步比较数学：我们需要把相机原点和方向向量都转过去
            
            # 相机原点在 world 下的坐标
            cam_origin = PointStamped()
            cam_origin.header = header
            cam_origin.point.x = 0.0
            cam_origin.point.y = 0.0
            cam_origin.point.z = 0.0
            p_origin_world = self.tf_buffer.transform(cam_origin, 'world')

            # 归一化点在 world 下的坐标
            p_vec_world = self.tf_buffer.transform(camera_point, 'world')

            # 5. 射线计算：利用相似三角形计算真实的 Z
            # 射线方程: P = Origin + t * (Vector - Origin)
            # 我们已知桌面高度 Z_table = self.table_height
            # 所以: P.z = Origin.z + t * (Vector.z - Origin.z) = table_height
            
            dz = p_vec_world.point.z - p_origin_world.point.z
            if abs(dz) < 1e-6:
                return # 射线平行于平面，无解

            t = (self.table_height - p_origin_world.point.z) / dz

            # 解出真实的 X 和 Y
            real_x = p_origin_world.point.x + t * (p_vec_world.point.x - p_origin_world.point.x)
            real_y = p_origin_world.point.y + t * (p_vec_world.point.y - p_origin_world.point.y)

            # --- C. 发布结果 ---
            target_msg = PointStamped()
            target_msg.header.frame_id = 'world'
            target_msg.header.stamp = self.get_clock().now().to_msg()
            target_msg.point.x = real_x
            target_msg.point.y = real_y
            target_msg.point.z = self.table_height
            
            self.target_pub.publish(target_msg)
            self.publish_marker(target_msg)
            
            # 日志输出，方便你写论文记录数据
            # self.get_logger().info(f"检测到香蕉! World坐标: X={real_x:.3f}, Y={real_y:.3f}, Z={self.table_height:.3f}")

        except Exception as e:
            self.get_logger().warn(f"TF 变换失败: {e}")

    def publish_marker(self, point_msg):
        # 在 RViz 里画一个黄色的球
        marker = Marker()
        marker.header = point_msg.header
        marker.ns = "banana"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = point_msg.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = BananaDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()