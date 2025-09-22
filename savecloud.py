#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import os
from datetime import datetime
import struct

class PointCloudSaver(Node):
    def __init__(self):
        super().__init__('pointcloud_saver')
        
        # Parameters
        self.declare_parameter('save_directory', 'pointclouds')
        self.declare_parameter('save_format', 'pcd')  # Options: 'pcd', 'ply', 'csv', 'npy'
        self.declare_parameter('max_clouds', -1)  # -1 for unlimited
        
        self.save_dir = self.get_parameter('save_directory').value
        self.save_format = self.get_parameter('save_format').value
        self.max_clouds = self.get_parameter('max_clouds').value
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Counter for saved clouds
        self.cloud_count = 0
        
        # Subscriber
        self.subscription = self.create_subscription(
            PointCloud2,
            '/segmentation_points',
            self.pointcloud_callback,
            10)
        
        self.get_logger().info(f'PointCloud Saver initialized. Saving to: {self.save_dir}')
        self.get_logger().info(f'Save format: {self.save_format}')
    
    def pointcloud_callback(self, msg):
        """Callback function for PointCloud2 messages"""
        if self.max_clouds > 0 and self.cloud_count >= self.max_clouds:
            self.get_logger().info(f'Reached maximum cloud count ({self.max_clouds}). Stopping...')
            return
        
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename_base = f'cloud_{self.cloud_count:04d}_{timestamp}'
            
            # Convert PointCloud2 to numpy array
            points = self.pointcloud2_to_array(msg)
            
            if points is None or len(points) == 0:
                self.get_logger().warn('Received empty point cloud')
                return
            
            # Save based on format
            if self.save_format == 'pcd':
                self.save_as_pcd(points, filename_base, msg)
            elif self.save_format == 'ply':
                self.save_as_ply(points, filename_base)
            elif self.save_format == 'csv':
                self.save_as_csv(points, filename_base)
            elif self.save_format == 'npy':
                self.save_as_numpy(points, filename_base)
            else:
                self.get_logger().error(f'Unknown save format: {self.save_format}')
                return
            
            self.cloud_count += 1
            self.get_logger().info(f'Saved cloud {self.cloud_count}: {filename_base}.{self.save_format}')
            
        except Exception as e:
            self.get_logger().error(f'Error saving point cloud: {str(e)}')
    
    def pointcloud2_to_array(self, cloud_msg):
        """Convert PointCloud2 message to numpy array"""
        try:
            # Extract points from PointCloud2 message
            points_list = []
            for point in pc2.read_points(cloud_msg, skip_nans=True):
                points_list.append(point)
            
            if not points_list:
                return None
            
            # Convert to numpy array
            points = np.array(points_list)
            return points
            
        except Exception as e:
            self.get_logger().error(f'Error converting PointCloud2: {str(e)}')
            return None
    
    def save_as_pcd(self, points, filename_base, msg):
        """Save point cloud in PCD format"""
        filename = os.path.join(self.save_dir, f'{filename_base}.pcd')
        
        with open(filename, 'w') as f:
            # Write PCD header
            f.write('# .PCD v0.7 - Point Cloud Data file format\n')
            f.write('VERSION 0.7\n')
            
            # Determine fields
            fields = []
            if points.shape[1] >= 3:
                fields.extend(['x', 'y', 'z'])
            if points.shape[1] >= 4:
                fields.append('intensity')
            if points.shape[1] >= 6:
                fields.extend(['rgb', 'normal_x', 'normal_y', 'normal_z'])
            
            f.write(f'FIELDS {" ".join(fields[:points.shape[1]])}\n')
            f.write(f'SIZE {" ".join(["4"] * points.shape[1])}\n')
            f.write(f'TYPE {" ".join(["F"] * points.shape[1])}\n')
            f.write(f'COUNT {" ".join(["1"] * points.shape[1])}\n')
            f.write(f'WIDTH {len(points)}\n')
            f.write('HEIGHT 1\n')
            f.write('VIEWPOINT 0 0 0 1 0 0 0\n')
            f.write(f'POINTS {len(points)}\n')
            f.write('DATA ascii\n')
            
            # Write point data
            for point in points:
                f.write(' '.join([str(p) for p in point]) + '\n')
    
    def save_as_ply(self, points, filename_base):
        """Save point cloud in PLY format"""
        filename = os.path.join(self.save_dir, f'{filename_base}.ply')
        
        with open(filename, 'w') as f:
            # Write PLY header
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(points)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            
            if points.shape[1] >= 4:
                f.write('property float intensity\n')
            if points.shape[1] >= 6:
                f.write('property uchar red\n')
                f.write('property uchar green\n')
                f.write('property uchar blue\n')
            
            f.write('end_header\n')
            
            # Write point data
            for point in points:
                f.write(' '.join([str(p) for p in point[:3]]))
                if points.shape[1] >= 4:
                    f.write(f' {point[3]}')
                if points.shape[1] >= 6:
                    # Assume RGB is packed in a single float
                    rgb = int(point[4])
                    r = (rgb >> 16) & 0xFF
                    g = (rgb >> 8) & 0xFF
                    b = rgb & 0xFF
                    f.write(f' {r} {g} {b}')
                f.write('\n')
    
    def save_as_csv(self, points, filename_base):
        """Save point cloud in CSV format"""
        filename = os.path.join(self.save_dir, f'{filename_base}.csv')
        
        # Determine headers
        headers = ['x', 'y', 'z']
        if points.shape[1] >= 4:
            headers.append('intensity')
        if points.shape[1] > 4:
            for i in range(4, points.shape[1]):
                headers.append(f'field_{i}')
        
        np.savetxt(filename, points, delimiter=',', 
                   header=','.join(headers), comments='')
    
    def save_as_numpy(self, points, filename_base):
        """Save point cloud as numpy array"""
        filename = os.path.join(self.save_dir, f'{filename_base}.npy')
        np.save(filename, points)


def main(args=None):
    rclpy.init(args=args)
    
    pointcloud_saver = PointCloudSaver()
    
    try:
        rclpy.spin(pointcloud_saver)
    except KeyboardInterrupt:
        pointcloud_saver.get_logger().info('Shutting down...')
    finally:
        pointcloud_saver.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()