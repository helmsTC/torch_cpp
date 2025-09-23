#!/usr/bin/env python3
"""
ROS2 node to read SemanticKITTI .bin files and publish as PointCloud2
Supports both single files and sequences
"""

import os
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct
from pathlib import Path
import glob
from datetime import datetime

class SemanticKITTIPublisher(Node):
    def __init__(self):
        super().__init__('semantickitti_publisher')
        
        # Declare parameters
        self.declare_parameter('input_path', '')  # Path to .bin file or directory
        self.declare_parameter('output_topic', '/lidar/world_pos')
        self.declare_parameter('frame_id', 'velodyne')
        self.declare_parameter('publish_rate', 10.0)  # Hz
        self.declare_parameter('loop', False)  # Loop sequence
        self.declare_parameter('load_labels', False)  # Load .label files if available
        self.declare_parameter('convert_to_cm', False)  # Convert from meters to cm
        self.declare_parameter('max_points', -1)  # -1 for no limit
        self.declare_parameter('min_range', 0.0)  # Minimum range filter
        self.declare_parameter('max_range', 100.0)  # Maximum range filter
        
        # Get parameters
        self.input_path = self.get_parameter('input_path').value
        self.output_topic = self.get_parameter('output_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.loop = self.get_parameter('loop').value
        self.load_labels = self.get_parameter('load_labels').value
        self.convert_to_cm = self.get_parameter('convert_to_cm').value
        self.max_points = self.get_parameter('max_points').value
        self.min_range = self.get_parameter('min_range').value
        self.max_range = self.get_parameter('max_range').value
        
        # Validate input path
        if not self.input_path:
            self.get_logger().error('No input path specified! Use --ros-args -p input_path:=/path/to/data')
            sys.exit(1)
        
        # Setup file list
        self.file_list = self.get_file_list()
        if not self.file_list:
            self.get_logger().error(f'No .bin files found at: {self.input_path}')
            sys.exit(1)
        
        self.current_index = 0
        
        # Create publisher
        self.publisher = self.create_publisher(
            PointCloud2,
            self.output_topic,
            10  # QoS depth
        )
        
        # Create timer for publishing
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_next_cloud
        )
        
        self.get_logger().info(f'SemanticKITTI Publisher initialized')
        self.get_logger().info(f'Input: {self.input_path}')
        self.get_logger().info(f'Found {len(self.file_list)} .bin files')
        self.get_logger().info(f'Publishing to: {self.output_topic}')
        self.get_logger().info(f'Rate: {self.publish_rate} Hz')
        self.get_logger().info(f'Units: {"centimeters" if self.convert_to_cm else "meters"}')
        
        # Statistics
        self.total_published = 0
        self.total_points = 0
    
    def get_file_list(self):
        """Get list of .bin files from input path"""
        input_path = Path(self.input_path)
        
        if input_path.is_file() and input_path.suffix == '.bin':
            # Single file
            return [str(input_path)]
        elif input_path.is_dir():
            # Directory - find all .bin files
            bin_files = sorted(glob.glob(str(input_path / '*.bin')))
            if not bin_files:
                # Try subdirectories (SemanticKITTI structure)
                bin_files = sorted(glob.glob(str(input_path / 'velodyne' / '*.bin')))
            return bin_files
        else:
            return []
    
    def load_bin_file(self, filepath):
        """Load point cloud from SemanticKITTI .bin file"""
        # Read binary file
        scan = np.fromfile(filepath, dtype=np.float32)
        
        # Reshape to N x 4 (x, y, z, intensity)
        if scan.shape[0] % 4 != 0:
            self.get_logger().warn(f'Invalid .bin file size: {filepath}')
            return None
        
        points = scan.reshape((-1, 4))
        
        # Apply range filter
        if self.min_range > 0 or self.max_range < 100:
            ranges = np.sqrt(np.sum(points[:, :3] ** 2, axis=1))
            mask = (ranges >= self.min_range) & (ranges <= self.max_range)
            points = points[mask]
        
        # Limit number of points if requested
        if self.max_points > 0 and points.shape[0] > self.max_points:
            # Random sampling
            indices = np.random.choice(points.shape[0], self.max_points, replace=False)
            points = points[indices]
        
        # Convert to centimeters if requested
        if self.convert_to_cm:
            points[:, :3] *= 100.0  # Convert m to cm
        
        return points
    
    def load_label_file(self, bin_filepath):
        """Load labels from .label file if it exists"""
        # Convert .bin path to .label path
        label_path = bin_filepath.replace('.bin', '.label')
        label_path = label_path.replace('velodyne', 'labels')
        
        if not os.path.exists(label_path):
            return None, None
        
        # Load label file
        label_data = np.fromfile(label_path, dtype=np.uint32)
        
        # Extract semantic and instance labels
        # Upper 16 bits: semantic label
        # Lower 16 bits: instance ID
        semantic_labels = label_data >> 16
        instance_labels = label_data & 0xFFFF
        
        # Apply SemanticKITTI learning map (optional)
        # This maps the raw labels to the reduced set used for training
        semantic_labels = self.apply_learning_map(semantic_labels)
        
        return semantic_labels, instance_labels
    
    def apply_learning_map(self, labels):
        """Apply SemanticKITTI learning map to reduce label set"""
        # SemanticKITTI learning map (simplified version)
        learning_map = {
            0: 0,     # "unlabeled"
            1: 0,     # "outlier" -> "unlabeled"
            10: 1,    # "car"
            11: 2,    # "bicycle"
            13: 5,    # "bus" -> "other-vehicle"
            15: 3,    # "motorcycle"
            16: 5,    # "on-rails" -> "other-vehicle"
            18: 4,    # "truck"
            20: 5,    # "other-vehicle"
            30: 6,    # "person"
            31: 7,    # "bicyclist"
            32: 8,    # "motorcyclist"
            40: 9,    # "road"
            44: 10,   # "parking"
            48: 11,   # "sidewalk"
            49: 12,   # "other-ground"
            50: 13,   # "building"
            51: 14,   # "fence"
            52: 0,    # "other-structure" -> "unlabeled"
            60: 9,    # "lane-marking" -> "road"
            70: 15,   # "vegetation"
            71: 16,   # "trunk"
            72: 17,   # "terrain"
            80: 18,   # "pole"
            81: 19,   # "traffic-sign"
            99: 0,    # "other-object" -> "unlabeled"
            252: 1,   # "moving-car" -> "car"
            253: 7,   # "moving-bicyclist" -> "bicyclist"
            254: 6,   # "moving-person" -> "person"
            255: 8,   # "moving-motorcyclist" -> "motorcyclist"
            256: 5,   # "moving-on-rails" -> "other-vehicle"
            257: 5,   # "moving-bus" -> "other-vehicle"
            258: 4,   # "moving-truck" -> "truck"
            259: 5,   # "moving-other-vehicle" -> "other-vehicle"
        }
        
        mapped_labels = np.zeros_like(labels)
        for raw, mapped in learning_map.items():
            mapped_labels[labels == raw] = mapped
        
        return mapped_labels
    
    def create_pointcloud2_msg(self, points, semantic_labels=None, instance_labels=None):
        """Create PointCloud2 message from numpy array"""
        msg = PointCloud2()
        
        # Header
        msg.header = Header()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Setup fields
        fields = []
        
        # XYZ fields
        fields.append(PointField(
            name='x', offset=0, datatype=PointField.FLOAT32, count=1
        ))
        fields.append(PointField(
            name='y', offset=4, datatype=PointField.FLOAT32, count=1
        ))
        fields.append(PointField(
            name='z', offset=8, datatype=PointField.FLOAT32, count=1
        ))
        
        # Intensity field
        fields.append(PointField(
            name='intensity', offset=12, datatype=PointField.FLOAT32, count=1
        ))
        
        offset = 16
        
        # Optional semantic label field
        if semantic_labels is not None:
            fields.append(PointField(
                name='semantic', offset=offset, datatype=PointField.UINT16, count=1
            ))
            offset += 2
        
        # Optional instance label field
        if instance_labels is not None:
            fields.append(PointField(
                name='instance', offset=offset, datatype=PointField.UINT16, count=1
            ))
            offset += 2
        
        msg.fields = fields
        
        # Set dimensions
        msg.height = 1
        msg.width = points.shape[0]
        msg.point_step = offset  # Size of each point in bytes
        msg.row_step = msg.point_step * msg.width
        msg.is_bigendian = False
        msg.is_dense = True
        
        # Create data buffer
        buffer = []
        
        for i in range(points.shape[0]):
            # Pack XYZ and intensity
            buffer.append(struct.pack('ffff',
                points[i, 0], points[i, 1], points[i, 2], points[i, 3]
            ))
            
            # Pack labels if available
            if semantic_labels is not None:
                buffer.append(struct.pack('H', int(semantic_labels[i])))
            
            if instance_labels is not None:
                buffer.append(struct.pack('H', int(instance_labels[i])))
        
        # Combine all data
        msg.data = b''.join(buffer)
        
        return msg
    
    def publish_next_cloud(self):
        """Publish the next point cloud in the sequence"""
        if self.current_index >= len(self.file_list):
            if self.loop:
                self.current_index = 0
                self.get_logger().info('Looping back to first file')
            else:
                self.get_logger().info(f'Finished publishing all {len(self.file_list)} files')
                self.timer.cancel()
                return
        
        # Load current file
        filepath = self.file_list[self.current_index]
        points = self.load_bin_file(filepath)
        
        if points is None:
            self.get_logger().error(f'Failed to load: {filepath}')
            self.current_index += 1
            return
        
        # Optionally load labels
        semantic_labels = None
        instance_labels = None
        if self.load_labels:
            semantic_labels, instance_labels = self.load_label_file(filepath)
            if semantic_labels is not None:
                # Filter labels to match filtered points
                if semantic_labels.shape[0] != points.shape[0]:
                    self.get_logger().warn(
                        f'Label count mismatch: {semantic_labels.shape[0]} vs {points.shape[0]} points'
                    )
                    semantic_labels = None
                    instance_labels = None
        
        # Create and publish message
        msg = self.create_pointcloud2_msg(points, semantic_labels, instance_labels)
        self.publisher.publish(msg)
        
        # Update statistics
        self.total_published += 1
        self.total_points += points.shape[0]
        avg_points = self.total_points / self.total_published
        
        # Log progress
        filename = os.path.basename(filepath)
        self.get_logger().info(
            f'[{self.current_index + 1}/{len(self.file_list)}] '
            f'Published {filename}: {points.shape[0]} points '
            f'(avg: {avg_points:.0f})'
        )
        
        self.current_index += 1
    
    def get_info_string(self):
        """Get information string about the node state"""
        return (
            f"SemanticKITTI Publisher Status:\n"
            f"  Files: {self.current_index}/{len(self.file_list)}\n"
            f"  Total published: {self.total_published}\n"
            f"  Total points: {self.total_points}\n"
            f"  Output topic: {self.output_topic}\n"
            f"  Frame ID: {self.frame_id}\n"
            f"  Rate: {self.publish_rate} Hz"
        )


def main(args=None):
    rclpy.init(args=args)
    
    node = SemanticKITTIPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Print final statistics
        node.get_logger().info('\nShutting down...')
        node.get_logger().info(node.get_info_string())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
