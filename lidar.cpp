#include "lidar/lidar.h"
#include <memory>
#include <chrono>

namespace lidar_processing
{

LidarProcessor::LidarProcessor(const rclcpp::NodeOptions &options)
    : Node("lidar_processor", options),
      model_loaded_(false)
{
    // Declare and get parameters
    this->declare_parameter<std::string>("model_path", "");
    this->declare_parameter<std::string>("input_topic", "/points");
    this->declare_parameter<bool>("use_cuda", false);
    this->declare_parameter<int>("batch_size", 1);
    
    model_path_ = this->get_parameter("model_path").as_string();
    std::string input_topic = this->get_parameter("input_topic").as_string();
    use_cuda_ = this->get_parameter("use_cuda").as_bool();
    batch_size_ = this->get_parameter("batch_size").as_int();
    
    // Check CUDA availability
    if (use_cuda_ && !torch::cuda::is_available()) {
        RCLCPP_WARN(this->get_logger(), "CUDA requested but not available. Using CPU instead.");
        use_cuda_ = false;
    }
    
    device_ = use_cuda_ ? torch::kCUDA : torch::kCPU;
    RCLCPP_INFO(this->get_logger(), "Using device: %s", use_cuda_ ? "CUDA" : "CPU");
    
    // Load the model
    if (!model_path_.empty()) {
        loadModel(model_path_);
    } else {
        RCLCPP_WARN(this->get_logger(), "No model path provided. Set 'model_path' parameter to load a model.");
    }
    
    // Create subscriber
    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic,
        rclcpp::SensorDataQoS(),
        std::bind(&LidarProcessor::pointcloudCallback, this, std::placeholders::_1));
    
    // Create publisher for processed data (example: could be modified based on your needs)
    processed_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/processed_points", 
        rclcpp::SensorDataQoS());
    
    RCLCPP_INFO(this->get_logger(), "LidarProcessor node initialized. Subscribing to: %s", input_topic.c_str());
}

bool LidarProcessor::loadModel(const std::string &model_path)
{
    try {
        // Deserialize the ScriptModule from file
        model_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path, device_));
        
        if (model_ == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load model from: %s", model_path.c_str());
            return false;
        }
        
        // Set to evaluation mode
        model_->eval();
        
        // Optionally freeze the model for inference
        model_ = std::make_shared<torch::jit::script::Module>(torch::jit::freeze(*model_));
        
        model_loaded_ = true;
        RCLCPP_INFO(this->get_logger(), "Successfully loaded model from: %s", model_path.c_str());
        
        // Test the model with dummy input
        testModel();
        
        return true;
    } catch (const c10::Error& e) {
        RCLCPP_ERROR(this->get_logger(), "Error loading model: %s", e.what());
        model_loaded_ = false;
        return false;
    }
}

void LidarProcessor::testModel()
{
    if (!model_loaded_) {
        RCLCPP_WARN(this->get_logger(), "Model not loaded, cannot test.");
        return;
    }
    
    try {
        // Create a dummy input tensor (adjust dimensions based on your model)
        // Example: batch_size x num_points x features (xyz + intensity)
        torch::Tensor dummy_input = torch::randn({1, 1000, 4}, device_);
        
        RCLCPP_INFO(this->get_logger(), "Testing model with input shape: [%ld, %ld, %ld]", 
                    dummy_input.size(0), dummy_input.size(1), dummy_input.size(2));
        
        // Forward pass
        auto output = model_->forward({dummy_input});
        
        RCLCPP_INFO(this->get_logger(), "Model test successful! Output type: %s", 
                    output.type_name());
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Model test failed: %s", e.what());
    }
}

void LidarProcessor::pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    
    RCLCPP_DEBUG(this->get_logger(), "Received pointcloud with %d points", 
                 msg->width * msg->height);
    
    // Convert PointCloud2 to tensor
    torch::Tensor point_tensor = pointcloud2ToTensor(msg);
    
    if (!point_tensor.defined() || point_tensor.numel() == 0) {
        RCLCPP_WARN(this->get_logger(), "Failed to convert pointcloud to tensor");
        return;
    }
    
    // Process with model if loaded
    if (model_loaded_) {
        torch::Tensor processed = processWithModel(point_tensor);
        
        // Convert back to PointCloud2 and publish (example)
        sensor_msgs::msg::PointCloud2 output_msg = tensorToPointcloud2(processed, msg->header);
        processed_pub_->publish(output_msg);
    } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                             "Model not loaded. Skipping processing.");
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    RCLCPP_DEBUG(this->get_logger(), "Processing took %ld ms", duration.count());
}

torch::Tensor LidarProcessor::pointcloud2ToTensor(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    // Get field offsets
    int x_offset = -1, y_offset = -1, z_offset = -1, intensity_offset = -1;
    
    for (const auto& field : msg->fields) {
        if (field.name == "x") x_offset = field.offset;
        else if (field.name == "y") y_offset = field.offset;
        else if (field.name == "z") z_offset = field.offset;
        else if (field.name == "intensity") intensity_offset = field.offset;
    }
    
    if (x_offset == -1 || y_offset == -1 || z_offset == -1) {
        RCLCPP_ERROR(this->get_logger(), "PointCloud2 missing required xyz fields");
        return torch::Tensor();
    }
    
    // Calculate number of points
    size_t num_points = msg->width * msg->height;
    int point_step = msg->point_step;
    
    // Determine number of features
    int num_features = 3;  // xyz
    if (intensity_offset != -1) {
        num_features = 4;  // xyz + intensity
    }
    
    // Create tensor
    torch::Tensor points = torch::zeros({static_cast<long>(num_points), num_features}, 
                                        torch::kFloat32);
    
    // Fill tensor with point data
    auto points_accessor = points.accessor<float, 2>();
    
    for (size_t i = 0; i < num_points; ++i) {
        const uint8_t* point_data = &msg->data[i * point_step];
        
        // Extract xyz
        points_accessor[i][0] = *reinterpret_cast<const float*>(point_data + x_offset);
        points_accessor[i][1] = *reinterpret_cast<const float*>(point_data + y_offset);
        points_accessor[i][2] = *reinterpret_cast<const float*>(point_data + z_offset);
        
        // Extract intensity if available
        if (intensity_offset != -1 && num_features == 4) {
            points_accessor[i][3] = *reinterpret_cast<const float*>(point_data + intensity_offset);
        }
    }
    
    // Move to appropriate device
    points = points.to(device_);
    
    // Add batch dimension
    points = points.unsqueeze(0);
    
    return points;
}

torch::Tensor LidarProcessor::processWithModel(const torch::Tensor &input)
{
    if (!model_loaded_) {
        RCLCPP_WARN(this->get_logger(), "Model not loaded");
        return input;
    }
    
    try {
        // Disable gradient computation for inference
        torch::NoGradGuard no_grad;
        
        // Forward pass through the model
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        
        auto output = model_->forward(inputs);
        
        // Handle different output types
        if (output.isTensor()) {
            return output.toTensor();
        } else if (output.isTuple()) {
            auto tuple = output.toTuple();
            if (tuple->elements().size() > 0 && tuple->elements()[0].isTensor()) {
                return tuple->elements()[0].toTensor();
            }
        }
        
        RCLCPP_WARN(this->get_logger(), "Unexpected model output type");
        return input;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error during model inference: %s", e.what());
        return input;
    }
}

sensor_msgs::msg::PointCloud2 LidarProcessor::tensorToPointcloud2(
    const torch::Tensor &tensor, 
    const std_msgs::msg::Header &header)
{
    sensor_msgs::msg::PointCloud2 msg;
    msg.header = header;
    
    // Remove batch dimension if present
    torch::Tensor points = tensor;
    if (points.dim() == 3 && points.size(0) == 1) {
        points = points.squeeze(0);
    }
    
    // Move to CPU if necessary
    if (points.is_cuda()) {
        points = points.cpu();
    }
    
    // Convert to contiguous float tensor
    points = points.contiguous().to(torch::kFloat32);
    
    size_t num_points = points.size(0);
    size_t num_features = points.size(1);
    
    // Set up message fields
    msg.height = 1;
    msg.width = num_points;
    msg.is_bigendian = false;
    msg.is_dense = true;
    
    // Define fields
    sensor_msgs::msg::PointField field_x;
    field_x.name = "x";
    field_x.offset = 0;
    field_x.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_x.count = 1;
    msg.fields.push_back(field_x);
    
    sensor_msgs::msg::PointField field_y;
    field_y.name = "y";
    field_y.offset = 4;
    field_y.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_y.count = 1;
    msg.fields.push_back(field_y);
    
    sensor_msgs::msg::PointField field_z;
    field_z.name = "z";
    field_z.offset = 8;
    field_z.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_z.count = 1;
    msg.fields.push_back(field_z);
    
    msg.point_step = 12;  // 3 * sizeof(float)
    
    if (num_features >= 4) {
        sensor_msgs::msg::PointField field_intensity;
        field_intensity.name = "intensity";
        field_intensity.offset = 12;
        field_intensity.datatype = sensor_msgs::msg::PointField::FLOAT32;
        field_intensity.count = 1;
        msg.fields.push_back(field_intensity);
        msg.point_step = 16;  // 4 * sizeof(float)
    }
    
    msg.row_step = msg.point_step * msg.width;
    msg.data.resize(msg.row_step * msg.height);
    
    // Copy data
    auto points_accessor = points.accessor<float, 2>();
    
    for (size_t i = 0; i < num_points; ++i) {
        uint8_t* point_data = &msg.data[i * msg.point_step];
        
        *reinterpret_cast<float*>(point_data + 0) = points_accessor[i][0];  // x
        *reinterpret_cast<float*>(point_data + 4) = points_accessor[i][1];  // y
        *reinterpret_cast<float*>(point_data + 8) = points_accessor[i][2];  // z
        
        if (num_features >= 4) {
            *reinterpret_cast<float*>(point_data + 12) = points_accessor[i][3];  // intensity
        }
    }
    
    return msg;
}

}  // namespace lidar_processing

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(lidar_processing::LidarProcessor)