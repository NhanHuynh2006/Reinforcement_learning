import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    package_name = 'articubot_one'

    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')

    # Default to TG_3 world in this package
    default_world = os.path.join(
        get_package_share_directory(package_name),
        'worlds',
        'TG_3.world'
    )

    # Ensure Gazebo can find ROS 2 plugins
    gazebo_plugin_path = [
        '/opt/ros/humble/lib',
        ':',
        EnvironmentVariable('GAZEBO_PLUGIN_PATH', default_value='')
    ]

    # robot_state_publisher (uses sim time)
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory(package_name), 'launch', 'rsp.launch.py')
        ]),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # Gazebo server/client with selected world
    gzserver = ExecuteProcess(
        cmd=[
            'gzserver',
            world,
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so',
        ],
        output='screen',
        additional_env={'GAZEBO_PLUGIN_PATH': gazebo_plugin_path},
    )

    gzclient = ExecuteProcess(
        cmd=['gzclient', '--gui-client-plugin=libgazebo_ros_eol_gui.so'],
        output='screen',
        additional_env={'GAZEBO_PLUGIN_PATH': gazebo_plugin_path},
    )

    # Spawn robot from robot_description
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'my_bot'],
        output='screen'
    )

    # Controllers for cmd_vel driving
    joint_broad_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_broad', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    diff_drive_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diff_cont', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use sim time if true'
        ),
        DeclareLaunchArgument(
            'world',
            default_value=default_world,
            description='Full path to world file'
        ),
        rsp,
        gzserver,
        gzclient,
        spawn_entity,
        joint_broad_spawner,
        diff_drive_spawner,
    ])
