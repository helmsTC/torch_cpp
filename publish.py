#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import struct

class PointCloud2Publisher(Node):
    def __init__(self):
        super().__init__('pointcloud2_publisher')
        
        # Create publisher
        self.publisher = self.create_publisher(PointCloud2, 'pointcloud', 10)
        
        # Create timer for publishing
        self.timer = self.create_timer(1.0, self.publish_pointcloud)
        
        self.get_logger().info('PointCloud2 publisher initialized')

    def create_point_cloud2_xyzi(self, points, intensities, frame_id='map'):
        """
        Create a PointCloud2 message with XYZ and intensity data.
        
        Args:
            points: numpy array of shape (N, 3) containing x, y, z coordinates
            intensities: numpy array of shape (N,) containing intensity values
            frame_id: frame ID for the point cloud
        
        Returns:
            PointCloud2 message
        """
        msg = PointCloud2()
        
        # Header
        msg.header = Header()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Define fields
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Set dimensions
        msg.height = 1  # unordered point cloud
        msg.width = len(points)
        msg.point_step = 16  # 4 fields * 4 bytes each
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.is_bigendian = False
        
        # Pack data
        data = []
        for i in range(len(points)):
            # Pack x, y, z, intensity as float32
            data.append(struct.pack('ffff', 
                                  float(points[i][0]), 
                                  float(points[i][1]), 
                                  float(points[i][2]), 
                                  float(intensities[i])))
        
        # Convert to bytes
        msg.data = b''.join(data)
        
        return msg

    def create_point_cloud2_xyzi_numpy(self, points, intensities, frame_id='map'):
        """
        Alternative method using numpy for better performance with large point clouds.
        """
        msg = PointCloud2()
        
        # Header
        msg.header = Header()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Define fields
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Set dimensions
        msg.height = 1
        msg.width = len(points)
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.is_bigendian = False
        
        # Create structured array
        cloud_data = np.zeros(len(points), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32)
        ])
        
        cloud_data['x'] = points[:, 0]
        cloud_data['y'] = points[:, 1]
        cloud_data['z'] = points[:, 2]
        cloud_data['intensity'] = intensities
        
        # Convert to bytes
        msg.data = cloud_data.tobytes()
        
        return msg

    def publish_pointcloud(self):
        """Publish sample point cloud data."""
        # Generate sample data - a spinning helix with varying intensity
        num_points = 1000
        t = np.linspace(0, 4 * np.pi, num_points)
        
        # Create helix shape
        radius = 2.0
        points = np.zeros((num_points, 3))
        points[:, 0] = radius * np.cos(t)  # x
        points[:, 1] = radius * np.sin(t)  # y
        points[:, 2] = t / 2.0              # z
        
        # Create intensity values (0-255 range, but can be any range)
        intensities = 100 + 50 * np.sin(t)
        
        # Create and publish message
        msg = self.create_point_cloud2_xyzi_numpy(points, intensities, 'map')
        self.publisher.publish(msg)
        
        self.get_logger().info(f'Published point cloud with {num_points} points')

def main(args=None):
    rclpy.init(args=args)
    
    node = PointCloud2Publisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()