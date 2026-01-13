import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, TimerAction, LogInfo
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
from launch.actions import (
    DeclareLaunchArgument,
    TimerAction,)
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)


def generate_launch_description():
     # =========================================================================
    # 1. 配置 (修改部分)
    # =========================================================================
    
    # 注意：MoveItConfigsBuilder 需要在 Python 解析时就确定路径，所以这里不能用 LaunchConfiguration
    # 请确保这里的包名和文件名是正确的
    DESCRIPTION_PACKAGE = "openarm_moveit_config"
    # 注意：这里如果你的 xacro 文件名有后缀，必须写全，例如 "openarm.urdf.xacro"
    # 如果你的文件名只是 "openarm_moveit_config"，请确认它真的是文件而不是文件夹
    DESCRIPTION_FILE = "mujoco_openarm.urdf.xacro" 
    
    # 使用 Python 原生方式获取路径 (返回的是字符串)
    pkg_share_path = get_package_share_directory(DESCRIPTION_PACKAGE)
    xacro_file_path = os.path.join(pkg_share_path, "config", DESCRIPTION_FILE)
    # print(f"_______________________{xacro_file_path}____________________")

    # mujoco_model_path = "/home/ldp/openarm_ws/openarm_mujoco/v1/openarm_bimanual_cam.xml"
    mujoco_model_path = "/home/ldp/openarm_ws/openarm_mujoco/v1/scene.xml"
    # mujoco_model_path = "/home/ldp/openarm_ws/openarm_mujoco/v0.3/scene.xml"

    # 这里的 controllers_file 可以保留用 LaunchConfiguration，因为它是在 Node 运行时才用的
    # 但为了统一和避免混乱，建议如果是同一个包下的，也直接用 os.path.join
    MOVEITCONTROLLERS_FILE = "moveit_controllers.yaml"
    moveit_controller_config_file_path = os.path.join(pkg_share_path, "config", MOVEITCONTROLLERS_FILE)
    MUJOCO_CONTROLLERS_FILE = "ros2_controllers.yaml"
    mujoco_controller_config_file_path = os.path.join(pkg_share_path, "config", MUJOCO_CONTROLLERS_FILE)
    # print(f"______________________________{mujoco_controller_config_file_path}")


    # =========================================================================
    # 生成 MoveIt 配置 (传入的是字符串路径)
    # =========================================================================
    moveit_config = MoveItConfigsBuilder("openarm", package_name=DESCRIPTION_PACKAGE) \
        .robot_description(file_path=xacro_file_path) \
        .trajectory_execution(file_path=moveit_controller_config_file_path) \
        .to_moveit_configs()

    use_sim_time = {"use_sim_time": True}
    TIMEOUT_SEC = "60"

    # =========================================================================
    # 2. 基础节点
    # =========================================================================
    
    # 静态 TF (World -> Base)
    node_static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link', '--ros-args', '-p', 'use_sim_time:=true'],
    )

    # MuJoCo
    node_mujoco = Node(
        package='mujoco_ros2_control',
        executable='mujoco_ros2_control',
        output='screen',
        parameters=[
            moveit_config.robot_description,
            mujoco_controller_config_file_path,
            use_sim_time,
            {'mujoco_model_path': mujoco_model_path}
        ]
    )

    # Robot State Publisher
    node_rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[moveit_config.robot_description, use_sim_time]
    )

    # Move Group
    node_move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            use_sim_time,
            {
                "publish_robot_description_semantic": True,
                "moveit_manage_controllers": True,  # 确保这里是 True
                "monitor_dynamics": False,
                # 新增以下参数，允许执行稍微慢一点，防止超时报错
                "trajectory_execution.allowed_execution_duration_scaling": 2.0,
                "trajectory_execution.allowed_goal_duration_margin": 0.5,
                "trajectory_execution.allowed_start_tolerance": 0.5,
            },
        ],
    )

    # RViz
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
            use_sim_time,
        ],
    )

    # =========================================================================
    # 3. 显式定义各个控制器 Spawner (注意逗号！)
    # =========================================================================
    
    # 1. 关节状态广播器
    spawner_jsb = Node(
        package="controller_manager",
        executable="spawner",
        name="spawner_jsb",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", "60",
            "--switch-timeout", "60" # <--- 逗号加上了，参数生效
        ],
        output="screen",
        # parameters=[{'use_sim_time': True}], # <--- 添加这一行
    )

    # 2. 左臂
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
        # parameters=[{'use_sim_time': True}], # <--- 添加这一行
    )

    # 3. 右臂
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
        # parameters=[{'use_sim_time': True}], # <--- 添加这一行
    )

    # 4. 左手
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
        # parameters=[{'use_sim_time': True}], # <--- 添加这一行
    )

    # 5. 右手
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
        # parameters=[{'use_sim_time': True}], # <--- 添加这一行
    )

    # =========================================================================
    # 4. 链式启动逻辑 (Chain Reaction)
    # =========================================================================
    
    # 基础节点列表
    nodes = [
        node_static_tf,
        node_mujoco,
        node_rsp,
        node_move_group,
        node_rviz,
    ]

    # 步骤 1: MuJoCo 启动 10秒后 -> 启动 JSB
    delay_jsb = TimerAction(
        period=10.0,
        actions=[
            LogInfo(msg="============= 正在启动 Joint State Broadcaster ============="),
            spawner_jsb
        ]
    )
    
    nodes.append(
        RegisterEventHandler(
            event_handler=OnProcessStart(
                target_action=node_mujoco,
                on_start=[delay_jsb]
            )
        )
    )

    # 步骤 2: JSB 退出(成功)后 -> 启动 左臂
    nodes.append(
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawner_jsb,
                on_exit=[
                    LogInfo(msg="============= JSB 完成，正在启动 左臂 ============="),
                    spawner_left_arm
                ]
            )
        )
    )

    # 步骤 3: 左臂 退出后 -> 启动 右臂
    nodes.append(
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawner_left_arm,
                on_exit=[
                    LogInfo(msg="============= 左臂 完成，正在启动 右臂 ============="),
                    spawner_right_arm
                ]
            )
        )
    )

    # 步骤 4: 右臂 退出后 -> 启动 左手
    nodes.append(
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawner_right_arm,
                on_exit=[
                    LogInfo(msg="============= 右臂 完成，正在启动 左手 ============="),
                    spawner_left_hand
                ]
            )
        )
    )

    # 步骤 5: 左手 退出后 -> 启动 右手
    nodes.append(
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawner_left_hand,
                on_exit=[
                    LogInfo(msg="============= 左手 完成，正在启动 右手 ============="),
                    spawner_right_hand
                ]
            )
        )
    )

    return LaunchDescription(nodes)