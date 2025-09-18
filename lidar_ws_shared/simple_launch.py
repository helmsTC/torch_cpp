#!/usr/bin/env python3
"""
Simplified launch file for MaskPLS lidar processor
No complex conditions - just launches the node
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate simple launch description"""
    
    # Declare parameters
    declare_model_path = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to the converted .pt model file'
    )
    
    declare_config_path = DeclareLaunchArgument(
        'config_path',
        default_value='',
        description='Path to config file (optional)'
    )
    
    declare_input_topic = DeclareLaunchArgument(
        'input_topic',
        default_value='/velodyne_points',
        description='Input PointCloud2 topic'
    )
    
    declare_use_cuda = DeclareLaunchArgument(
        'use_cuda',
        default_value='true',
        description='Use CUDA if available'
    )
    
    declare_verbose = DeclareLaunchArgument(
        'verbose',
        default_value='false',
        description='Enable verbose logging'
    )
    
    declare_publish_markers = DeclareLaunchArgument(
        'publish_markers',
        default_value='true',
        description='Publish visualization markers'
    )
    
    declare_auto_start = DeclareLaunchArgument(
        'auto_start_server',
        default_value='true',
        description='Auto-start Python server'
    )
    
    # Create the node
    lidar_node = Node(
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
            'num_classes': 20,
            'overlap_threshold': 0.8,
            'x_limits': [-48.0, 48.0],
            'y_limits': [-48.0, 48.0],
            'z_limits': [-4.0, 1.5],
            'things_ids': [1, 2, 3, 4, 5, 6, 7, 8],
            'min_points_per_instance': 50,
            'max_instances': 100,
        }],
        remappings=[
            ('points', LaunchConfiguration('input_topic')),
        ]
    )
    
    # Log message
    log_msg = LogInfo(msg='Starting MaskPLS Lidar Processor (Simplified Launch)')
    
    return LaunchDescription([
        declare_model_path,
        declare_config_path,
        declare_input_topic,
        declare_use_cuda,
        declare_verbose,
        declare_publish_markers,
        declare_auto_start,
        log_msg,
        lidar_node
    ])
