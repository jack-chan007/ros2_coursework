import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import cv2
import numpy as np
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

class BananaDetector(Node):
    def __init__(self):
        super().__init__('banana_detector')
        
        # 1. 订阅
        self.img_sub = self.create_subscription(Image, '/depth_camera/image_raw', self.image_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/depth_camera/camera_info', self.info_callback, 10)
        self.target_pub = self.create_publisher(PointStamped, '/banana_position', 10)
        self.marker_pub = self.create_publisher(Marker, '/banana_marker', 10)
        
        self.bridge = CvBridge()
        self.camera_model = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # --- 【关键参数调整区】 ---
        
        # 1. 桌面高度：XML中桌子高0.4，厚0.02 => 表面0.41。
        # 香蕉有厚度，所以抓取中心大约在 0.425 ~ 0.43 左右
        self.table_height = 0.425  

        # 2. 【核心修正】手动误差补偿 (单位：米)
        # 建议先归零，看看原始误差是多少，然后再微调
        self.offset_x = 0.04   # 前后偏移
        self.offset_y = -0.13   # 左右偏移

        self.get_logger().info(f" 🍌  检测启动 | 补偿参数: X={self.offset_x}, Y={self.offset_y}")
        self.get_logger().info(" 📺  可视化窗口已开启: Banana Detection View")

    def info_callback(self, msg):
        if self.camera_model is None:
            self.camera_model = msg

    def image_callback(self, msg):
        if self.camera_model is None: return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception: return

        # HSV 颜色识别
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 默认显示原图，如果有检测结果会在下面画上去
        display_img = cv_image.copy()

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # --- 【新增】可视化绘制 ---
                # 画出绿色轮廓
                cv2.drawContours(display_img, [c], -1, (0, 255, 0), 2)
                # 画出红色中心点
                cv2.circle(display_img, (cX, cY), 7, (0, 0, 255), -1)
                # 写文字
                cv2.putText(display_img, "Banana", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                self.process_coordinates(cX, cY, msg.header)

        # --- 【新增】显示图像窗口 ---
        cv2.imshow("Banana Detection View", display_img)
        cv2.waitKey(1)

    def process_coordinates(self, u, v, header):
        # 1. 像素 -> 归一化相机坐标
        fx = self.camera_model.k[0]
        fy = self.camera_model.k[4]
        cx = self.camera_model.k[2]
        cy = self.camera_model.k[5]
        
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy

        camera_point = PointStamped()
        camera_point.header = header
        camera_point.point.x = x_norm
        camera_point.point.y = y_norm
        camera_point.point.z = 1.0

        try:
            # 2. 获取 TF
            if not self.tf_buffer.can_transform('world', header.frame_id, rclpy.time.Time()):
                return
                
            # 3. 射线投影 (Ray Casting)
            # A. 相机原点
            origin_pt = PointStamped()
            origin_pt.header = header
            origin_world = self.tf_buffer.transform(origin_pt, 'world')
            
            # B. 指向点
            vec_world = self.tf_buffer.transform(camera_point, 'world')
            
            # C. 计算射线与平面的交点
            dz = vec_world.point.z - origin_world.point.z
            if abs(dz) < 1e-6: return
            
            t = (self.table_height - origin_world.point.z) / dz
            
            raw_x = origin_world.point.x + t * (vec_world.point.x - origin_world.point.x)
            raw_y = origin_world.point.y + t * (vec_world.point.y - origin_world.point.y)

            # 4. 【应用手动补偿】
            final_x = raw_x + self.offset_x
            final_y = raw_y + self.offset_y

            # 发布
            target_msg = PointStamped()
            target_msg.header.frame_id = 'world'
            target_msg.header.stamp = self.get_clock().now().to_msg()
            target_msg.point.x = final_x
            target_msg.point.y = final_y
            target_msg.point.z = self.table_height

            self.target_pub.publish(target_msg)
            self.publish_marker(target_msg)
            
            # 降频打印日志，方便调试
            self.get_logger().info(f"抓取点: X={final_x:.3f}, Y={final_y:.3f} (Offset: {self.offset_x}, {self.offset_y})", throttle_duration_sec=2.0)

        except Exception as e:
            pass

    def publish_marker(self, point_msg):
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
    
    # --- 【新增】关闭窗口 ---
    cv2.destroyAllWindows()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
