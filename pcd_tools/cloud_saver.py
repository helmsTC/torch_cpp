#!/usr/bin/env python3
"""
ROS2 node to save semantic point clouds from /semantic_points topic
Supports saving in multiple formats: PCD, PLY, NPZ
"""

import os
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
from datetime import datetime
import struct
from pathlib import Path
import open3d as o3d  # Optional, for PCD/PLY support

class SemanticPointCloudSaver(Node):
    def __init__(self):
        super().__init__('semantic_cloud_saver')
        
        # Declare parameters
        self.declare_parameter('save_dir', './saved_clouds')
        self.declare_parameter('save_format', 'pcd')  # Options: 'npz', 'pcd', 'ply', 'all'
        self.declare_parameter('save_interval', 1)  # Save every N clouds
        self.declare_parameter('max_saves', -1)  # -1 for unlimited
        self.declare_parameter('include_timestamp', True)
        self.declare_parameter('save_raw_binary', False)  # For debugging
        
        # Get parameters
        self.save_dir = self.get_parameter('save_dir').value
        self.save_format = self.get_parameter('save_format').value
        self.save_interval = self.get_parameter('save_interval').value
        self.max_saves = self.get_parameter('max_saves').value
        self.include_timestamp = self.get_parameter('include_timestamp').value
        self.save_raw_binary = self.get_parameter('save_raw_binary').value
        
        # Create save directory
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize counters
        self.cloud_count = 0
        self.saved_count = 0
        
        # Subscribe to semantic points
        self.subscription = self.create_subscription(
            PointCloud2,
            '/semantic_points',
            self.cloud_callback,
            10
        )
        
        self.get_logger().info(f'Semantic cloud saver initialized')
        self.get_logger().info(f'Saving to: {self.save_dir}')
        self.get_logger().info(f'Format: {self.save_format}')
        self.get_logger().info(f'Save interval: every {self.save_interval} clouds')
        
        # Check if open3d is available for PCD/PLY formats
        self.has_open3d = self.check_open3d()
        if not self.has_open3d and self.save_format in ['pcd', 'ply', 'all']:
            self.get_logger().warn('Open3D not available, will use NPZ format instead')
            if self.save_format != 'all':
                self.save_format = 'npz'
    
    def check_open3d(self):
        """Check if Open3D is available"""
        try:
            import open3d
            return True
        except ImportError:
            return False
    
    def cloud_callback(self, msg):
        """Callback for received point clouds"""
        self.cloud_count += 1
        
        # Check if we should save this cloud
        if self.cloud_count % self.save_interval != 0:
            return
        
        # Check max saves limit
        if self.max_saves > 0 and self.saved_count >= self.max_saves:
            self.get_logger().info(f'Reached max saves limit ({self.max_saves}), stopping')
            rclpy.shutdown()
            return
        
        try:
            # Parse the point cloud
            cloud_data = self.parse_pointcloud2(msg)
            
            if cloud_data is None:
                self.get_logger().error('Failed to parse point cloud')
                return
            
            # Generate filename base
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3] if self.include_timestamp else ''
            base_name = f'cloud_{self.saved_count:06d}'
            if timestamp:
                base_name = f'{base_name}_{timestamp}'
            
            # Save in requested format(s)
            if self.save_format == 'all':
                self.save_npz(cloud_data, base_name)
                if self.has_open3d:
                    self.save_pcd(cloud_data, base_name)
                    self.save_ply(cloud_data, base_name)
            elif self.save_format == 'npz':
                self.save_npz(cloud_data, base_name)
            elif self.save_format == 'pcd' and self.has_open3d:
                self.save_pcd(cloud_data, base_name)
            elif self.save_format == 'ply' and self.has_open3d:
                self.save_ply(cloud_data, base_name)
            else:
                self.save_npz(cloud_data, base_name)
            
            # Optionally save raw binary for debugging
            if self.save_raw_binary:
                self.save_raw(msg, base_name)
            
            self.saved_count += 1
            
            # Log statistics
            num_points = cloud_data['points'].shape[0]
            unique_labels = len(np.unique(cloud_data['semantic_labels']))
            unique_instances = len(np.unique(cloud_data['instance_labels']))
            
            self.get_logger().info(
                f'Saved cloud {self.saved_count}: {num_points} points, '
                f'{unique_labels} semantic classes, {unique_instances} instances'
            )
            
        except Exception as e:
            self.get_logger().error(f'Error processing cloud: {e}')
            import traceback
            traceback.print_exc()
    
    def parse_pointcloud2(self, msg):
        """Parse PointCloud2 message into numpy arrays"""
        try:
            # Get field information
            fields = {}
            for field in msg.fields:
                fields[field.name] = {
                    'offset': field.offset,
                    'datatype': field.datatype,
                    'count': field.count
                }
            
            # Check required fields
            if 'x' not in fields or 'y' not in fields or 'z' not in fields:
                self.get_logger().error('Missing XYZ fields in point cloud')
                return None
            
            # Parse points
            points = []
            colors = []
            semantic_labels = []
            instance_labels = []
            
            point_step = msg.point_step
            data = msg.data
            
            for i in range(msg.width * msg.height):
                offset = i * point_step
                
                # Extract XYZ
                x = struct.unpack('f', data[offset + fields['x']['offset']:
                                          offset + fields['x']['offset'] + 4])[0]
                y = struct.unpack('f', data[offset + fields['y']['offset']:
                                          offset + fields['y']['offset'] + 4])[0]
                z = struct.unpack('f', data[offset + fields['z']['offset']:
                                          offset + fields['z']['offset'] + 4])[0]
                points.append([x, y, z])
                
                # Extract RGB if available
                if 'rgb' in fields:
                    rgb_packed = struct.unpack('f', data[offset + fields['rgb']['offset']:
                                                       offset + fields['rgb']['offset'] + 4])[0]
                    rgb_int = struct.unpack('I', struct.pack('f', rgb_packed))[0]
                    r = (rgb_int >> 16) & 0xFF
                    g = (rgb_int >> 8) & 0xFF
                    b = rgb_int & 0xFF
                    colors.append([r, g, b])
                
                # Extract label if available (combined semantic and instance)
                if 'label' in fields:
                    label = struct.unpack('I', data[offset + fields['label']['offset']:
                                                   offset + fields['label']['offset'] + 4])[0]
                    semantic_label = (label >> 16) & 0xFFFF
                    instance_label = label & 0xFFFF
                    semantic_labels.append(semantic_label)
                    instance_labels.append(instance_label)
            
            # Convert to numpy arrays
            result = {
                'points': np.array(points, dtype=np.float32),
                'header': {
                    'frame_id': msg.header.frame_id,
                    'stamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                }
            }
            
            if colors:
                result['colors'] = np.array(colors, dtype=np.uint8)
            
            if semantic_labels:
                result['semantic_labels'] = np.array(semantic_labels, dtype=np.int32)
            
            if instance_labels:
                result['instance_labels'] = np.array(instance_labels, dtype=np.int32)
            
            return result
            
        except Exception as e:
            self.get_logger().error(f'Error parsing PointCloud2: {e}')
            return None
    
    def save_npz(self, cloud_data, base_name):
        """Save cloud data as compressed NumPy archive"""
        filepath = os.path.join(self.save_dir, f'{base_name}.npz')
        np.savez_compressed(filepath, **cloud_data)
        self.get_logger().debug(f'Saved NPZ: {filepath}')
    
    def save_pcd(self, cloud_data, base_name):
        """Save as PCD file using Open3D"""
        if not self.has_open3d:
            return
        
        try:
            import open3d as o3d
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud_data['points'])
            
            # Add colors if available
            if 'colors' in cloud_data:
                pcd.colors = o3d.utility.Vector3dVector(cloud_data['colors'] / 255.0)
            elif 'semantic_labels' in cloud_data:
                # Generate colors from semantic labels
                colors = self.labels_to_colors(cloud_data['semantic_labels'])
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Save PCD
            filepath = os.path.join(self.save_dir, f'{base_name}.pcd')
            o3d.io.write_point_cloud(filepath, pcd, write_ascii=False)
            self.get_logger().debug(f'Saved PCD: {filepath}')
            
        except Exception as e:
            self.get_logger().error(f'Error saving PCD: {e}')
    
    def save_ply(self, cloud_data, base_name):
        """Save as PLY file using Open3D"""
        if not self.has_open3d:
            return
        
        try:
            import open3d as o3d
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud_data['points'])
            
            # Add colors if available
            if 'colors' in cloud_data:
                pcd.colors = o3d.utility.Vector3dVector(cloud_data['colors'] / 255.0)
            elif 'semantic_labels' in cloud_data:
                # Generate colors from semantic labels
                colors = self.labels_to_colors(cloud_data['semantic_labels'])
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Save PLY
            filepath = os.path.join(self.save_dir, f'{base_name}.ply')
            o3d.io.write_point_cloud(filepath, pcd, write_ascii=False)
            self.get_logger().debug(f'Saved PLY: {filepath}')
            
        except Exception as e:
            self.get_logger().error(f'Error saving PLY: {e}')
    
    def save_raw(self, msg, base_name):
        """Save raw PointCloud2 message data for debugging"""
        filepath = os.path.join(self.save_dir, f'{base_name}_raw.bin')
        with open(filepath, 'wb') as f:
            f.write(msg.data)
        
        # Also save metadata
        meta_filepath = os.path.join(self.save_dir, f'{base_name}_meta.txt')
        with open(meta_filepath, 'w') as f:
            f.write(f'Frame ID: {msg.header.frame_id}\n')
            f.write(f'Timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}\n')
            f.write(f'Height: {msg.height}\n')
            f.write(f'Width: {msg.width}\n')
            f.write(f'Point step: {msg.point_step}\n')
            f.write(f'Row step: {msg.row_step}\n')
            f.write(f'Is dense: {msg.is_dense}\n')
            f.write(f'Is bigendian: {msg.is_bigendian}\n')
            f.write('\nFields:\n')
            for field in msg.fields:
                f.write(f'  {field.name}: offset={field.offset}, '
                       f'datatype={field.datatype}, count={field.count}\n')
    
    def labels_to_colors(self, labels):
        """Convert semantic labels to colors"""
        # Simple color mapping - you can customize this
        max_label = labels.max() if labels.size > 0 else 1
        colors = np.zeros((labels.shape[0], 3))
        
        for i in range(labels.shape[0]):
            label = labels[i]
            # Generate color using HSV colorspace
            hue = (label * 0.618033988749895) % 1.0  # Golden ratio
            colors[i] = self.hsv_to_rgb(hue, 0.8, 0.9)
        
        return colors
    
    def hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB"""
        h_i = int(h * 6)
        f = h * 6 - h_i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        if h_i == 0:
            return [v, t, p]
        elif h_i == 1:
            return [q, v, p]
        elif h_i == 2:
            return [p, v, t]
        elif h_i == 3:
            return [p, q, v]
        elif h_i == 4:
            return [t, p, v]
        else:
            return [v, p, q]


def main(args=None):
    rclpy.init(args=args)
    
    saver_node = SemanticPointCloudSaver()
    
    try:
        rclpy.spin(saver_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Print summary
        saver_node.get_logger().info(
            f'Shutting down. Saved {saver_node.saved_count} clouds '
            f'out of {saver_node.cloud_count} received.'
        )
        saver_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
