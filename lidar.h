#ifndef LIDAR_PROCESSOR_H
#define LIDAR_PROCESSOR_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <array>

namespace lidar_processing
{

// Structure to hold semantic class info
struct SemanticClass {
    std::string name;
    std::array<float, 3> color;  // RGB values normalized to [0, 1]
};

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
    void initializeSemanticMapping();
    
    // Processing functions
    std::pair<torch::Tensor, torch::Tensor> processWithModel(const torch::Tensor &points, const torch::Tensor &features);
    torch::Tensor prepareModelInput(const torch::Tensor &points, const torch::Tensor &features);
    
    // Conversion utilities
    std::pair<torch::Tensor, torch::Tensor> pointcloud2ToTensor(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    sensor_msgs::msg::PointCloud2 createLabeledPointCloud(
        const torch::Tensor &points,
        const torch::Tensor &semantic_labels,
        const torch::Tensor &instance_labels,
        const std_msgs::msg::Header &header);
    
    // Panoptic inference (mimics Python implementation)
    std::pair<torch::Tensor, torch::Tensor> panopticInference(
        const torch::Tensor &pred_logits,
        const torch::Tensor &pred_masks,
        const torch::Tensor &padding);
    
    // Helper functions
    std::array<uint8_t, 3> getColorForClass(int class_id);
    std::array<uint8_t, 3> getInstanceColor(int instance_id);
    void filterPointsByBounds(torch::Tensor &points, torch::Tensor &features);
    
    // ROS2 interfaces
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr semantic_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr panoptic_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    
    // Model related
    std::shared_ptr<torch::jit::script::Module> model_;
    std::string model_path_;
    bool model_loaded_;
    
    // Model configuration
    int num_classes_;
    std::vector<int> things_ids_;
    float overlap_threshold_;
    
    // Processing parameters
    torch::Device device_ = torch::kCPU;
    bool use_cuda_;
    int batch_size_;
    
    // Spatial bounds for KITTI
    std::array<float, 2> x_limits_;
    std::array<float, 2> y_limits_;
    std::array<float, 2> z_limits_;
    float voxel_resolution_;
    
    // Semantic-KITTI mapping
    std::unordered_map<int, SemanticClass> semantic_map_;
    std::unordered_map<int, int> learning_map_;
    std::unordered_map<int, int> learning_map_inv_;
    
    // Performance monitoring
    std::chrono::steady_clock::time_point last_inference_time_;
    double avg_inference_time_;
    int inference_count_;
};

}  // namespace lidar_processing

#endif  // LIDAR_PROCESSOR_H
