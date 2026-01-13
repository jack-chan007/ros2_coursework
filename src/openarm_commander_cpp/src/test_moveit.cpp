#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <thread>
#include <memory>

int main(int argc, char **argv)
{
    // 初始化 ROS 2 节点
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("test_moveit");
    
    // 设置日志级别（可选，便于调试）
    rclcpp::Logger logger = node->get_logger();
    RCLCPP_INFO(logger, "===== Test MoveIt Planning Start =====");

    // 创建单线程执行器并启动自旋线程
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    auto spinner = std::thread([&executor]() { 
        try {
            executor.spin(); 
        } catch (const std::exception &e) {
            RCLCPP_ERROR(rclcpp::get_logger("spinner"), "Executor spin failed: %s", e.what());
        }
    });

    try {
        // 初始化 MoveGroupInterface（左机械臂规划组）
        auto left_arm = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node, "left_arm");
        left_arm->setMaxVelocityScalingFactor(1.0);
        left_arm->setMaxAccelerationScalingFactor(1.0);

        // 设置规划起始状态为当前状态
        left_arm->setStartStateToCurrentState();
        // 设置命名目标（需确保 SRDF 中已定义 "hand_up" 目标位姿）
        left_arm->setNamedTarget("hand_up");
        // 增加规划超时时间（默认5s，可根据机械臂复杂度调整）
        left_arm->setPlanningTime(10.0);

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        moveit::core::MoveItErrorCode plan_result = left_arm->plan(plan);

        if (plan_result == moveit::core::MoveItErrorCode::SUCCESS)
        {
            RCLCPP_INFO(logger, "Planning SUCCESS! Executing trajectory...");
            // 执行规划好的轨迹
            moveit::core::MoveItErrorCode exec_result = left_arm->execute(plan);
            if (exec_result == moveit::core::MoveItErrorCode::SUCCESS) {
                RCLCPP_INFO(logger, "Execution SUCCESS!");
            } else {
                RCLCPP_ERROR(logger, "Execution FAILED! Error code: %d", exec_result.val);
            }
        }
        else
        {
            // ========== 补充的 else 分支核心逻辑 ==========
            RCLCPP_ERROR(logger, "Planning FAILED! Error code: %d", plan_result.val);
            
            // 1. 打印错误码对应的含义（便于定位问题）
            switch (plan_result.val) {
                case moveit::core::MoveItErrorCode::PLANNING_FAILED:
                    RCLCPP_ERROR(logger, "Reason: Planning algorithm failed to find a valid path");
                    break;
                case moveit::core::MoveItErrorCode::INVALID_MOTION_PLAN:
                    RCLCPP_ERROR(logger, "Reason: Invalid motion plan (e.g. target out of joint limits)");
                    break;
                case moveit::core::MoveItErrorCode::TIMED_OUT:
                    RCLCPP_ERROR(logger, "Reason: Planning timeout (increase planning time with setPlanningTime())");
                    break;
                case moveit::core::MoveItErrorCode::INVALID_ROBOT_STATE:
                    RCLCPP_ERROR(logger, "Reason: Invalid robot start state");
                    break;
                default:
                    RCLCPP_ERROR(logger, "Reason: Unknown error (check kinematics/config/SRDF)");
                    break;
            }

            // 2. 打印调试信息（帮助排查问题）
            RCLCPP_WARN(logger, "=== Debug Info ===");
            RCLCPP_WARN(logger, "Planning group: %s", left_arm->getName().c_str());
            RCLCPP_WARN(logger, "Named target: hand_up");
            RCLCPP_WARN(logger, "Current joint state:");
            auto current_joints = left_arm->getCurrentJointValues();
            auto joint_names = left_arm->getJointNames();
            for (size_t i = 0; i < joint_names.size(); ++i) {
                RCLCPP_WARN(logger, "  %s: %.3f", joint_names[i].c_str(), current_joints[i]);
            }

            // 3. 可选：尝试回退到初始位姿（容错处理）
            RCLCPP_WARN(logger, "Trying to move back to 'home' position...");
            left_arm->setNamedTarget("home"); // 需确保 SRDF 中定义 "home" 位姿
            moveit::planning_interface::MoveGroupInterface::Plan fallback_plan;
            if (left_arm->plan(fallback_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
                left_arm->execute(fallback_plan);
                RCLCPP_WARN(logger, "Fallback to home position SUCCESS");
            } else {
                RCLCPP_ERROR(logger, "Fallback to home position FAILED");
            }
        }
    } catch (const std::exception &e) {
        // 捕获 MoveIt 接口可能抛出的异常
        RCLCPP_FATAL(logger, "MoveIt interface exception: %s", e.what());
    }

    // 资源清理
    RCLCPP_INFO(logger, "===== Test MoveIt Planning End =====");
    rclcpp::shutdown();
    if (spinner.joinable()) {
        spinner.join();
    }

    return 0;
}