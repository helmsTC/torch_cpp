#!/usr/bin/env python3
"""
Launch file for the MaskPLS lidar processor with shared memory
This launches both the C++ ROS node and the Python inference server
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('lidar_processor')
    
    # Declare launch arguments
    declare_args = [
        DeclareLaunchArgument(
            'model_path',
            default_value='',
            description='Path to the converted .pt model file'
        ),
        DeclareLaunchArgument(
            'config_path',
            default_value='',
            description='Path to config file (optional)'
        ),
        DeclareLaunchArgument(
            'input_topic',
            default_value='/velodyne_points',
            description='Input PointCloud2 topic'
        ),
        DeclareLaunchArgument(
            'use_cuda',
            default_value='true',
            description='Use CUDA if available'
        ),
        DeclareLaunchArgument(
            'auto_start_server',
            default_value='true',
            description='Automatically start Python inference server'
        ),
        DeclareLaunchArgument(
            'publish_markers',
            default_value='true',
            description='Publish instance visualization markers'
        ),
        DeclareLaunchArgument(
            'verbose',
            default_value='false',
            description='Enable verbose logging'
        ),
        DeclareLaunchArgument(
            'num_classes',
            default_value='20',
            description='Number of semantic classes (20 for KITTI)'
        ),
        DeclareLaunchArgument(
            'overlap_threshold',
            default_value='0.8',
            description='Overlap threshold for instance segmentation'
        ),
        DeclareLaunchArgument(
            'x_min',
            default_value='-48.0',
            description='Minimum x boundary'
        ),
        DeclareLaunchArgument(
            'x_max',
            default_value='48.0',
            description='Maximum x boundary'
        ),
        DeclareLaunchArgument(
            'y_min',
            default_value='-48.0',
            description='Minimum y boundary'
        ),
        DeclareLaunchArgument(
            'y_max',
            default_value='48.0',
            description='Maximum y boundary'
        ),
        DeclareLaunchArgument(
            'z_min',
            default_value='-4.0',
            description='Minimum z boundary'
        ),
        DeclareLaunchArgument(
            'z_max',
            default_value='1.5',
            description='Maximum z boundary'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Logging level (debug, info, warn, error)'
        ),
        DeclareLaunchArgument(
            'start_python_server',
            default_value='false',
            description='Start Python server as separate process (if auto_start_server is false)'
        ),
    ]
    
    # Create the C++ ROS node
    lidar_processor_node = Node(
        package='lidar_processor',
        executable='lidar_processor_shm_node',
        name='lidar_processor_shm',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'config_path': LaunchConfiguration('config_path'),
            'input_topic': LaunchConfiguration('input_topic'),
            'use_cuda': LaunchConfiguration('use_cuda'),
            'auto_start_server': LaunchConfiguration('auto_start_server'),
            'publish_markers': LaunchConfiguration('publish_markers'),
            'verbose': LaunchConfiguration('verbose'),
            'num_classes': LaunchConfiguration('num_classes'),
            'overlap_threshold': LaunchConfiguration('overlap_threshold'),
            'x_limits': [
                LaunchConfiguration('x_min'),
                LaunchConfiguration('x_max')
            ],
            'y_limits': [
                LaunchConfiguration('y_min'),
                LaunchConfiguration('y_max')
            ],
            'z_limits': [
                LaunchConfiguration('z_min'),
                LaunchConfiguration('z_max')
            ],
        }],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        remappings=[
            ('points', LaunchConfiguration('input_topic')),
        ]
    )
    
    # Optionally start Python inference server as separate process
    # (useful for debugging or if auto_start_server is false)
    python_server_cmd = [
        'python3',
        os.path.join(pkg_dir, 'lib', 'lidar_processor', 'maskpls_inference_server.py'),
        '--model', LaunchConfiguration('model_path')
    ]
    
    # Add optional arguments
    python_server = ExecuteProcess(
        cmd=python_server_cmd,
        output='screen',
        condition=PythonExpression([
            "'", LaunchConfiguration('start_python_server'), "' == 'true' and '",
            LaunchConfiguration('model_path'), "' != ''"
        ])
    )
    
    # Log info about the configuration
    log_info = LogInfo(
        msg=['Starting MaskPLS Lidar Processor with shared memory'])
    
    log_model = LogInfo(
        msg=['Model path: ', LaunchConfiguration('model_path')],
        condition=PythonExpression([
            "'", LaunchConfiguration('model_path'), "' != ''"
        ])
    )
    
    return LaunchDescription([
        *declare_args,
        log_info,
        log_model,
        python_server,
        lidar_processor_node,
    ])
