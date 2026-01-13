import os
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess, RegisterEventHandler, TimerAction, LogInfo,
    DeclareLaunchArgument, IncludeLaunchDescription
)
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.substitutions import Command, PathJoinSubstitution, LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # =========================================================================
    # 1. 基础配置 (适配 Gazebo Classic 11)
    # =========================================================================
    DESCRIPTION_PACKAGE = "openarm_moveit_config"
    DESCRIPTION_FILE = "openarm.gazebo.urdf.xacro" 
    TIMEOUT_SEC = "60"

    # 获取包路径（Python 原生方式）
    pkg_share_path = get_package_share_directory(DESCRIPTION_PACKAGE)
    xacro_file_path = os.path.join(pkg_share_path, "config", DESCRIPTION_FILE)
    moveit_controller_config_file_path = os.path.join(pkg_share_path, "config", "moveit_controllers.yaml")
    mujoco_controller_config_file_path = os.path.join(pkg_share_path, "config", "ros2_controllers.yaml")

    # 定义启动参数
    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument(
        "use_sim_time",
        default_value=TextSubstitution(text="true"),
        description="Use simulation time (Gazebo)"
    ))
    ld.add_action(DeclareLaunchArgument(
        "gazebo_world",
        default_value=PathJoinSubstitution([
            get_package_share_directory('gazebo_ros'), "worlds", "empty.world"
        ]),
        description="Gazebo Classic world file path"
    ))

    # =========================================================================
    # 2. 生成 MoveIt 配置
    # =========================================================================
    moveit_config = MoveItConfigsBuilder("openarm", package_name=DESCRIPTION_PACKAGE) \
        .robot_description(file_path=xacro_file_path) \
        .trajectory_execution(file_path=moveit_controller_config_file_path) \
        .to_moveit_configs()

    # =========================================================================
    # 3. 替换 ros_gz_sim 为 Gazebo Classic 启动逻辑 (核心修改)
    # =========================================================================
    # 3.1 启动 Gazebo Classic 服务器 (gzserver)
    gazebo_server = ExecuteProcess(
        cmd=[
            "gzserver",
            "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so",
            LaunchConfiguration("gazebo_world"),
        ],
        output="screen",
        name="gazebo_server"
    )
    ld.add_action(gazebo_server)

    # 3.2 启动 Gazebo Classic 客户端 (gzclient)
    gazebo_client = ExecuteProcess(
        cmd=["gzclient"],
        output="screen",
        name="gazebo_client"
    )
    ld.add_action(gazebo_client)

    # 3.3 将机器人模型生成到 Gazebo Classic (替换 ros_gz_sim/create)
    spawn_robot_in_gazebo = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_robot",
        arguments=[
            "-topic", "/robot_description",
            "-entity", "openarm",
            "-x", "0.0", "-y", "0.0", "-z", "0.0"
        ],
        output="screen",
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}]
    )
    # 等待 Gazebo 服务器启动后再生成模型
    ld.add_action(RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=gazebo_server,
            on_start=[
                LogInfo(msg="============= Gazebo 服务器启动，生成机器人模型 ============="),
                spawn_robot_in_gazebo
            ]
        )
    ))

    # 3.4 Clock Bridge (适配 Gazebo Classic)
    clock_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}]
    )
    ld.add_action(clock_bridge_node)

    # =========================================================================
    # 4. 核心节点 (修复 use_sim_time 配置)
    # =========================================================================
    # 4.1 静态 TF (World -> Base)
    node_static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link'],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        name="static_tf_publisher"
    )
    ld.add_action(node_static_tf)

    # 4.2 Robot State Publisher
    node_rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name="robot_state_publisher",
        output='both',
        parameters=[
            moveit_config.robot_description, 
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
            {"publish_frequency": 30.0}
        ],
    )
    ld.add_action(node_rsp)

    # 4.3 Move Group (核心规划节点)
    node_move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
            {
                "publish_robot_description_semantic": True,
                "moveit_manage_controllers": True,
                "monitor_dynamics": False,
                "trajectory_execution.allowed_execution_duration_scaling": 2.0,
                "trajectory_execution.allowed_goal_duration_margin": 0.5,
                "trajectory_execution.allowed_start_tolerance": 0.5,
            },
        ],
        name="move_group"
    )
    ld.add_action(node_move_group)

    # 4.4 RViz2
    rviz_config = PathJoinSubstitution([FindPackageShare("openarm_moveit_config"), "config", "moveit.rviz"])
    node_rviz = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        name="rviz2"
    )
    ld.add_action(node_rviz)

    # =========================================================================
    # 5. 添加 ROS2 Control 节点 (关键缺失项！)
    # =========================================================================
    # 5.1 Controller Manager (硬件接口桥接 Gazebo)
    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        name="controller_manager",
        parameters=[
            moveit_config.robot_description,
            mujoco_controller_config_file_path,
            {"use_sim_time": LaunchConfiguration("use_sim_time")}
        ],
        output="screen"
    )
    ld.add_action(controller_manager)

    # =========================================================================
    # 6. 控制器 Spawner (修复启动逻辑，关联 Controller Manager)
    # =========================================================================
    # 6.1 关节状态广播器 (JSB)
    spawner_jsb = Node(
        package="controller_manager",
        executable="spawner",
        name="spawner_jsb",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", TIMEOUT_SEC,
            "--switch-timeout", TIMEOUT_SEC
        ],
        output="screen",
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    # 6.2 左臂控制器
    spawner_left_arm = Node(
        package="controller_manager",
        executable="spawner",
        name="spawner_left_arm",
        arguments=[
            "left_arm_controller",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", TIMEOUT_SEC,
            "--switch-timeout", TIMEOUT_SEC
        ],
        output="screen",
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    # 6.3 右臂控制器
    spawner_right_arm = Node(
        package="controller_manager",
        executable="spawner",
        name="spawner_right_arm",
        arguments=[
            "right_arm_controller",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", TIMEOUT_SEC,
            "--switch-timeout", TIMEOUT_SEC
        ],
        output="screen",
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    # 6.4 左手控制器
    spawner_left_hand = Node(
        package="controller_manager",
        executable="spawner",
        name="spawner_left_hand",
        arguments=[
            "left_hand_controller",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", TIMEOUT_SEC,
            "--switch-timeout", TIMEOUT_SEC
        ],
        output="screen",
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    # 6.5 右手控制器
    spawner_right_hand = Node(
        package="controller_manager",
        executable="spawner",
        name="spawner_right_hand",
        arguments=[
            "right_hand_controller",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", TIMEOUT_SEC,
            "--switch-timeout", TIMEOUT_SEC
        ],
        output="screen",
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    # =========================================================================
    # 7. 修复控制器链式启动逻辑 (关联 Controller Manager 启动)
    # =========================================================================
    # 步骤1: Controller Manager 启动后 5秒 -> 启动 JSB
    delay_jsb = TimerAction(
        period=5.0,
        actions=[
            LogInfo(msg="============= 启动 Joint State Broadcaster ============="),
            spawner_jsb
        ]
    )
    ld.add_action(RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=controller_manager,
            on_start=[delay_jsb]
        )
    ))

    # 步骤2: JSB 启动成功后 -> 启动左臂
    ld.add_action(RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawner_jsb,
            on_exit=[
                LogInfo(msg="============= JSB 启动完成，启动左臂控制器 ============="),
                spawner_left_arm
            ]
        )
    ))

    # 步骤3: 左臂启动后 -> 启动右臂
    ld.add_action(RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawner_left_arm,
            on_exit=[
                LogInfo(msg="============= 左臂控制器启动完成，启动右臂控制器 ============="),
                spawner_right_arm
            ]
        )
    ))

    # 步骤4: 右臂启动后 -> 启动左手
    ld.add_action(RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawner_right_arm,
            on_exit=[
                LogInfo(msg="============= 右臂控制器启动完成，启动左手控制器 ============="),
                spawner_left_hand
            ]
        )
    ))

    # 步骤5: 左手启动后 -> 启动右手
    ld.add_action(RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawner_left_hand,
            on_exit=[
                LogInfo(msg="============= 左手控制器启动完成，启动右手控制器 ============="),
                spawner_right_hand
            ]
        )
    ))

    return ld