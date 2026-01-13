import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from srdfdom.srdf import SRDF
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # -------------------------------------------------------------------------
    # 1. 构建 MoveIt 配置
    # -------------------------------------------------------------------------
    # MoveItConfigsBuilder 负责读取 URDF, SRDF, OMPL 等配置文件
    # 我们这里指定机器人名字为 "openarm"，包名为 "openarm_moveit_config"
    moveit_config = MoveItConfigsBuilder("openarm", package_name="openarm_moveit_config").to_moveit_configs()

    # -------------------------------------------------------------------------
    # 2. 定义启动参数 (Launch Arguments)
    # -------------------------------------------------------------------------
    ld = LaunchDescription()
    
    # 是否使用 RViz
    ld.add_action(DeclareLaunchArgument("use_rviz", default_value="true", description="Launch RViz"))
    
    # -------------------------------------------------------------------------
    # 3. 节点: Robot State Publisher (RSP)
    # -------------------------------------------------------------------------
    # 该节点发布机器人的 /tf 和 /robot_description
    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        respawn=True,
        output="screen",
        parameters=[
            moveit_config.robot_description,
            {"publish_frequency": 100.0},
        ],
    )
    ld.add_action(rsp_node)

    # -------------------------------------------------------------------------
    # 4. 节点: Static TF for Virtual Joints (虚拟关节)
    # -------------------------------------------------------------------------
    # 如果你在 SRDF 中定义了虚拟关节（例如 robot 连到 world），需要发布静态 TF
    # 这里通过解析 SRDF 文件内容自动生成 static_transform_publisher
    if moveit_config.robot_description_semantic:
        name_counter = 0
        for key, xml_contents in moveit_config.robot_description_semantic.items():
            srdf = SRDF.from_xml_string(xml_contents)
            for vj in srdf.virtual_joints:
                ld.add_action(
                    Node(
                        package="tf2_ros",
                        executable="static_transform_publisher",
                        name=f"static_transform_publisher{name_counter}",
                        output="log",
                        arguments=[
                            "--frame-id", vj.parent_frame,
                            "--child-frame-id", vj.child_link,
                        ],
                    )
                )
                name_counter += 1

    # -------------------------------------------------------------------------
    # 5. 节点: ROS 2 Control Node (硬件抽象层)
    # -------------------------------------------------------------------------
    # 在 demo 模式下，这是 fake/mock 硬件接口，负责接收控制指令并反馈关节状态
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            str(moveit_config.package_path / "config/ros2_controllers.yaml"),
        ],
        output="screen",
    )
    ld.add_action(ros2_control_node)

    # -------------------------------------------------------------------------
    # 6. 节点: Controller Spawners (控制器加载器)
    # -------------------------------------------------------------------------
    # 从 moveit_controllers.yaml 中读取控制器名称并生成 Spawner 节点
    # 通常包括 joint_state_broadcaster 和你的机械臂控制器(如 openarm_controller)
    
    # 获取控制器列表
    controller_names = moveit_config.trajectory_execution.get(
        "moveit_simple_controller_manager", {}
    ).get("controller_names", [])

    # 循环启动每个控制器
    for controller in controller_names + ["joint_state_broadcaster"]:
        ld.add_action(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[controller],
                output="screen",
            )
        )

    # -------------------------------------------------------------------------
    # 7. 节点: Move Group (核心规划节点)
    # -------------------------------------------------------------------------
    # 这是 MoveIt 的大脑，负责运动规划、碰撞检测等
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(), # 注入所有配置 (URDF, SRDF, Kinematics, Joint Limits 等)
            {
                "publish_robot_description_semantic": True,
                "allow_trajectory_execution": True, # 允许执行轨迹
                "moveit_manage_controllers": True,  # 让 MoveIt 管理控制器
                "publish_planning_scene": True,
                "publish_geometry_updates": True,
                "publish_state_updates": True,
                "publish_transforms_updates": True,
            },
        ],
    )
    ld.add_action(move_group_node)

    # -------------------------------------------------------------------------
    # 8. 节点: RViz (可视化)
    # -------------------------------------------------------------------------
    rviz_config_file = str(moveit_config.package_path / "config/moveit.rviz")
    
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
        ],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )
    ld.add_action(rviz_node)


    # node_mujoco_ros2_control = Node(
    #     package='mujoco_ros2_control',
    #     executable='mujoco_ros2_control',
    #     output='screen',
    #     parameters=[
    #         moveit_config.robot_description,
    #         controller_config_file,
    #         {'mujoco_model_path':os.path.join(mujoco_ros2_control_demos_path, 'mujoco_models', 'test_cart.xml')}
    #     ]
    # )


    return ld