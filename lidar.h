#ifndef LIDAR_PROCESSOR_H
#define LIDAR_PROCESSOR_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <string>
#include <vector>
#include <memory>

namespace lidar_processing
{

class LidarProcessor : public rclcpp::Node
{
public:
    explicit LidarProcessor(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
    ~LidarProcessor() = default;

private:
    // ROS2 callbacks
    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    
    // Model management
    bool loadModel(const std::string &model_path);
    void testModel();
    
    // Processing functions
    torch::Tensor processWithModel(const torch::Tensor &input);
    
    // Conversion utilities
    torch::Tensor pointcloud2ToTensor(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    sensor_msgs::msg::PointCloud2 tensorToPointcloud2(
        const torch::Tensor &tensor, 
        const std_msgs::msg::Header &header);
    
    // ROS2 interfaces
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr processed_pub_;
    
    // Model related
    std::shared_ptr<torch::jit::script::Module> model_;
    std::string model_path_;
    bool model_loaded_;
    
    // Processing parameters
    torch::Device device_ = torch::kCPU;
    bool use_cuda_;
    int batch_size_;
};

}  // namespace lidar_processing

#endif  // LIDAR_PROCESSOR_H