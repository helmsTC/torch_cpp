#!/usr/bin/env python3
"""
Launch file for the MaskPLS lidar processor node
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
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
            'batch_size',
            default_value='1',
            description='Batch size for inference'
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
            'voxel_resolution',
            default_value='0.05',
            description='Voxel resolution for the model'
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
    ]
    
    # Create the node
    lidar_processor_node = Node(
        package='lidar_processor',
        executable='lidar_processor_node',
        name='lidar_processor',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'input_topic': LaunchConfiguration('input_topic'),
            'use_cuda': LaunchConfiguration('use_cuda'),
            'batch_size': LaunchConfiguration('batch_size'),
            'num_classes': LaunchConfiguration('num_classes'),
            'overlap_threshold': LaunchConfiguration('overlap_threshold'),
            'voxel_resolution': LaunchConfiguration('voxel_resolution'),
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
    
    # Log info about the configuration
    log_info = LogInfo(
        msg=['Starting MaskPLS Lidar Processor with model: ', LaunchConfiguration('model_path')]
    )
    
    return LaunchDescription([
        *declare_args,
        log_info,
        lidar_processor_node,
    ])
