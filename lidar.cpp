#include "lidar/lidar.h"
#include <memory>
#include <chrono>
#include <random>
#include <pcl_conversions/pcl_conversions.h>

namespace lidar_processing
{

LidarProcessor::LidarProcessor(const rclcpp::NodeOptions &options)
    : Node("lidar_processor", options),
      model_loaded_(false),
      num_classes_(20),  // KITTI default
      overlap_threshold_(0.8),
      voxel_resolution_(0.05),
      avg_inference_time_(0.0),
      inference_count_(0)
{
    // Declare and get parameters
    this->declare_parameter<std::string>("model_path", "");
    this->declare_parameter<std::string>("input_topic", "/points");
    this->declare_parameter<bool>("use_cuda", false);
    this->declare_parameter<int>("batch_size", 1);
    this->declare_parameter<int>("num_classes", 20);
    this->declare_parameter<double>("overlap_threshold", 0.8);
    this->declare_parameter<double>("voxel_resolution", 0.05);
    
    // Spatial bounds parameters (KITTI defaults)
    this->declare_parameter<std::vector<double>>("x_limits", {-48.0, 48.0});
    this->declare_parameter<std::vector<double>>("y_limits", {-48.0, 48.0});
    this->declare_parameter<std::vector<double>>("z_limits", {-4.0, 1.5});
    
    model_path_ = this->get_parameter("model_path").as_string();
    std::string input_topic = this->get_parameter("input_topic").as_string();
    use_cuda_ = this->get_parameter("use_cuda").as_bool();
    batch_size_ = this->get_parameter("batch_size").as_int();
    num_classes_ = this->get_parameter("num_classes").as_int();
    overlap_threshold_ = this->get_parameter("overlap_threshold").as_double();
    voxel_resolution_ = this->get_parameter("voxel_resolution").as_double();
    
    // Get spatial bounds
    auto x_lim = this->get_parameter("x_limits").as_double_array();
    auto y_lim = this->get_parameter("y_limits").as_double_array();
    auto z_lim = this->get_parameter("z_limits").as_double_array();
    x_limits_ = {static_cast<float>(x_lim[0]), static_cast<float>(x_lim[1])};
    y_limits_ = {static_cast<float>(y_lim[0]), static_cast<float>(y_lim[1])};
    z_limits_ = {static_cast<float>(z_lim[0]), static_cast<float>(z_lim[1])};
    
    // Things IDs for KITTI
    things_ids_ = {1, 2, 3, 4, 5, 6, 7, 8};
    
    // Check CUDA availability
    if (use_cuda_ && !torch::cuda::is_available()) {
        RCLCPP_WARN(this->get_logger(), "CUDA requested but not available. Using CPU instead.");
        use_cuda_ = false;
    }
    
    device_ = use_cuda_ ? torch::kCUDA : torch::kCPU;
    RCLCPP_INFO(this->get_logger(), "Using device: %s", use_cuda_ ? "CUDA" : "CPU");
    
    // Initialize semantic mapping
    initializeSemanticMapping();
    
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
    
    // Create publishers
    semantic_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/semantic_points", rclcpp::SensorDataQoS());
    
    panoptic_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/panoptic_points", rclcpp::SensorDataQoS());
    
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/instance_markers", 10);
    
    RCLCPP_INFO(this->get_logger(), "LidarProcessor initialized. Subscribing to: %s", input_topic.c_str());
}

void LidarProcessor::initializeSemanticMapping()
{
    // Semantic-KITTI color mapping (BGR -> RGB, normalized)
    semantic_map_[0] = {"unlabeled", {0.0f, 0.0f, 0.0f}};
    semantic_map_[1] = {"car", {0.39f, 0.59f, 0.96f}};
    semantic_map_[2] = {"bicycle", {0.96f, 0.90f, 0.39f}};
    semantic_map_[3] = {"motorcycle", {0.12f, 0.24f, 0.59f}};
    semantic_map_[4] = {"truck", {0.31f, 0.12f, 0.71f}};
    semantic_map_[5] = {"other-vehicle", {0.0f, 0.0f, 1.0f}};
    semantic_map_[6] = {"person", {1.0f, 0.12f, 0.12f}};
    semantic_map_[7] = {"bicyclist", {1.0f, 0.16f, 0.78f}};
    semantic_map_[8] = {"motorcyclist", {0.59f, 0.12f, 0.35f}};
    semantic_map_[9] = {"road", {1.0f, 0.0f, 1.0f}};
    semantic_map_[10] = {"parking", {1.0f, 0.59f, 1.0f}};
    semantic_map_[11] = {"sidewalk", {0.29f, 0.0f, 0.29f}};
    semantic_map_[12] = {"other-ground", {0.29f, 0.0f, 0.69f}};
    semantic_map_[13] = {"building", {1.0f, 0.78f, 0.0f}};
    semantic_map_[14] = {"fence", {0.98f, 0.47f, 0.20f}};
    semantic_map_[15] = {"vegetation", {0.0f, 0.69f, 0.0f}};
    semantic_map_[16] = {"trunk", {0.53f, 0.24f, 0.0f}};
    semantic_map_[17] = {"terrain", {0.59f, 0.94f, 0.31f}};
    semantic_map_[18] = {"pole", {1.0f, 0.94f, 0.59f}};
    semantic_map_[19] = {"traffic-sign", {1.0f, 0.0f, 0.0f}};
    
    RCLCPP_INFO(this->get_logger(), "Initialized semantic mapping with %zu classes", semantic_map_.size());
}

bool LidarProcessor::loadModel(const std::string &model_path)
{
    try {
        // Load the TorchScript model
        model_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path, device_));
        
        if (model_ == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load model from: %s", model_path.c_str());
            return false;
        }
        
        // Set to evaluation mode
        model_->eval();
        
        model_loaded_ = true;
        RCLCPP_INFO(this->get_logger(), "Successfully loaded model from: %s", model_path.c_str());
        
        // Test the model
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
        // Create dummy input matching MaskPLS expected format
        // The model expects a batch dict with 'pt_coord' and 'feats'
        int num_points = 1000;
        torch::Tensor dummy_points = torch::randn({num_points, 3}, device_);
        torch::Tensor dummy_features = torch::randn({num_points, 4}, device_);
        
        // Create input dictionary (as IValue)
        c10::Dict<std::string, c10::IValue> batch_dict;
        batch_dict.insert("pt_coord", c10::List<torch::Tensor>({dummy_points}));
        batch_dict.insert("feats", c10::List<torch::Tensor>({dummy_features}));
        
        RCLCPP_INFO(this->get_logger(), "Testing model with %d points", num_points);
        
        // Forward pass
        auto output = model_->forward({batch_dict});
        
        RCLCPP_INFO(this->get_logger(), "Model test successful!");
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Model test failed: %s", e.what());
    }
}

void LidarProcessor::pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    if (!model_loaded_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                             "Model not loaded. Skipping processing.");
        return;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    RCLCPP_DEBUG(this->get_logger(), "Received pointcloud with %d points", 
                 msg->width * msg->height);
    
    // Convert PointCloud2 to tensors
    auto [points_tensor, features_tensor] = pointcloud2ToTensor(msg);
    
    if (!points_tensor.defined() || points_tensor.numel() == 0) {
        RCLCPP_WARN(this->get_logger(), "Failed to convert pointcloud to tensor");
        return;
    }
    
    // Process with model
    try {
        auto [semantic_pred, instance_pred] = processWithModel(points_tensor, features_tensor);
        
        // Create and publish labeled point cloud
        auto labeled_cloud = createLabeledPointCloud(
            points_tensor, semantic_pred, instance_pred, msg->header);
        
        semantic_pub_->publish(labeled_cloud);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Update performance stats
        inference_count_++;
        avg_inference_time_ = (avg_inference_time_ * (inference_count_ - 1) + duration.count()) / inference_count_;
        
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "Inference time: %ld ms (avg: %.2f ms)", 
                            duration.count(), avg_inference_time_);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error during processing: %s", e.what());
    }
}

std::pair<torch::Tensor, torch::Tensor> LidarProcessor::pointcloud2ToTensor(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    // Get field offsets
    int x_offset = -1, y_offset = -1, z_offset = -1, intensity_offset = -1;
    
    for (const auto& field : msg->fields) {
        if (field.name == "x") x_offset = field.offset;
        else if (field.name == "y") y_offset = field.offset;
        else if (field.name == "z") z_offset = field.offset;
        else if (field.name == "intensity" || field.name == "i") intensity_offset = field.offset;
    }
    
    if (x_offset == -1 || y_offset == -1 || z_offset == -1) {
        RCLCPP_ERROR(this->get_logger(), "PointCloud2 missing required xyz fields");
        return {torch::Tensor(), torch::Tensor()};
    }
    
    size_t num_points = msg->width * msg->height;
    int point_step = msg->point_step;
    
    // Create tensors for points and features
    torch::Tensor points = torch::zeros({static_cast<long>(num_points), 3}, torch::kFloat32);
    torch::Tensor features = torch::zeros({static_cast<long>(num_points), 4}, torch::kFloat32);
    
    auto points_accessor = points.accessor<float, 2>();
    auto features_accessor = features.accessor<float, 2>();
    
    // Extract point data
    for (size_t i = 0; i < num_points; ++i) {
        const uint8_t* point_data = &msg->data[i * point_step];
        
        float x = *reinterpret_cast<const float*>(point_data + x_offset);
        float y = *reinterpret_cast<const float*>(point_data + y_offset);
        float z = *reinterpret_cast<const float*>(point_data + z_offset);
        
        points_accessor[i][0] = x;
        points_accessor[i][1] = y;
        points_accessor[i][2] = z;
        
        // Features: xyz + intensity
        features_accessor[i][0] = x;
        features_accessor[i][1] = y;
        features_accessor[i][2] = z;
        
        if (intensity_offset != -1) {
            features_accessor[i][3] = *reinterpret_cast<const float*>(point_data + intensity_offset);
        } else {
            features_accessor[i][3] = 0.0f;  // Default intensity
        }
    }
    
    // Filter points by spatial bounds
    torch::Tensor mask = (points.select(1, 0) >= x_limits_[0]) & (points.select(1, 0) <= x_limits_[1]) &
                         (points.select(1, 1) >= y_limits_[0]) & (points.select(1, 1) <= y_limits_[1]) &
                         (points.select(1, 2) >= z_limits_[0]) & (points.select(1, 2) <= z_limits_[1]);
    
    points = points.index({mask});
    features = features.index({mask});
    
    // Move to device
    points = points.to(device_);
    features = features.to(device_);
    
    return {points, features};
}

std::pair<torch::Tensor, torch::Tensor> LidarProcessor::processWithModel(
    const torch::Tensor &points, const torch::Tensor &features)
{
    torch::NoGradGuard no_grad;
    
    // Create batch dictionary for model input
    c10::Dict<std::string, c10::IValue> batch_dict;
    batch_dict.insert("pt_coord", c10::List<torch::Tensor>({points.cpu()}));
    batch_dict.insert("feats", c10::List<torch::Tensor>({features.cpu()}));
    
    // Forward pass
    auto output = model_->forward({batch_dict});
    
    // Parse output - expecting tuple of (outputs_dict, padding, bb_logits)
    if (!output.isTuple()) {
        throw std::runtime_error("Expected tuple output from model");
    }
    
    auto output_tuple = output.toTuple();
    if (output_tuple->elements().size() < 2) {
        throw std::runtime_error("Model output tuple has insufficient elements");
    }
    
    // Get outputs dictionary and padding
    auto outputs_dict = output_tuple->elements()[0].toGenericDict();
    auto padding = output_tuple->elements()[1].toTensor();
    
    // Extract predictions
    auto pred_logits = outputs_dict.at("pred_logits").toTensor();
    auto pred_masks = outputs_dict.at("pred_masks").toTensor();
    
    // Perform panoptic inference
    auto [semantic_pred, instance_pred] = panopticInference(pred_logits, pred_masks, padding);
    
    return {semantic_pred, instance_pred};
}

std::pair<torch::Tensor, torch::Tensor> LidarProcessor::panopticInference(
    const torch::Tensor &pred_logits,
    const torch::Tensor &pred_masks,
    const torch::Tensor &padding)
{
    // This mimics the Python panoptic_inference function
    int batch_size = pred_logits.size(0);
    
    std::vector<torch::Tensor> sem_preds, ins_preds;
    
    for (int b = 0; b < batch_size; ++b) {
        auto mask_cls = pred_logits[b];
        auto mask_pred = pred_masks[b].index({~padding[b]}).sigmoid();
        
        // Get predictions
        auto [scores, labels] = mask_cls.max(-1);
        auto keep = labels.ne(num_classes_);
        
        if (keep.sum().item<int>() == 0) {
            // No valid predictions
            int num_points = mask_pred.size(0);
            sem_preds.push_back(torch::zeros({num_points}, torch::kInt32).to(device_));
            ins_preds.push_back(torch::zeros({num_points}, torch::kInt32).to(device_));
            continue;
        }
        
        auto cur_scores = scores.index({keep});
        auto cur_classes = labels.index({keep});
        auto cur_masks = mask_pred.index({"...", keep});
        
        // Probability masks
        auto cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks;
        
        // Initialize output tensors
        int num_points = cur_masks.size(0);
        auto semantic_seg = torch::zeros({num_points}, torch::kInt32).to(device_);
        auto instance_seg = torch::zeros({num_points}, torch::kInt32).to(device_);
        
        if (cur_masks.size(1) > 0) {
            auto cur_mask_ids = cur_prob_masks.argmax(1);
            
            int current_segment_id = 0;
            std::unordered_map<int, int> stuff_memory_list;
            
            for (int k = 0; k < cur_classes.size(0); ++k) {
                int pred_class = cur_classes[k].item<int>();
                
                // Check if it's a thing class
                bool is_thing = std::find(things_ids_.begin(), things_ids_.end(), pred_class) != things_ids_.end();
                
                auto mask_k = (cur_mask_ids == k) & (cur_masks.select(1, k) >= 0.5);
                int mask_area = mask_k.sum().item<int>();
                
                if (mask_area > 0) {
                    if (!is_thing) {
                        // Stuff class - merge regions
                        if (stuff_memory_list.find(pred_class) == stuff_memory_list.end()) {
                            current_segment_id++;
                            stuff_memory_list[pred_class] = current_segment_id;
                        }
                        instance_seg.index_put_({mask_k}, 0);
                    } else {
                        // Thing class - unique instance
                        current_segment_id++;
                        instance_seg.index_put_({mask_k}, current_segment_id);
                    }
                    semantic_seg.index_put_({mask_k}, pred_class);
                }
            }
        }
        
        sem_preds.push_back(semantic_seg);
        ins_preds.push_back(instance_seg);
    }
    
    // For single batch, return first element
    return {sem_preds[0].cpu(), ins_preds[0].cpu()};
}

sensor_msgs::msg::PointCloud2 LidarProcessor::createLabeledPointCloud(
    const torch::Tensor &points,
    const torch::Tensor &semantic_labels,
    const torch::Tensor &instance_labels,
    const std_msgs::msg::Header &header)
{
    sensor_msgs::msg::PointCloud2 msg;
    msg.header = header;
    
    // Ensure tensors are on CPU
    auto points_cpu = points.cpu().contiguous();
    auto sem_cpu = semantic_labels.cpu().contiguous();
    auto ins_cpu = instance_labels.cpu().contiguous();
    
    int num_points = points_cpu.size(0);
    
    // Setup message structure with XYZRGB
    msg.height = 1;
    msg.width = num_points;
    msg.is_bigendian = false;
    msg.is_dense = true;
    
    // Define fields for XYZRGB
    sensor_msgs::msg::PointField field_x, field_y, field_z, field_rgb;
    
    field_x.name = "x";
    field_x.offset = 0;
    field_x.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_x.count = 1;
    
    field_y.name = "y";
    field_y.offset = 4;
    field_y.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_y.count = 1;
    
    field_z.name = "z";
    field_z.offset = 8;
    field_z.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_z.count = 1;
    
    field_rgb.name = "rgb";
    field_rgb.offset = 12;
    field_rgb.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_rgb.count = 1;
    
    msg.fields = {field_x, field_y, field_z, field_rgb};
    msg.point_step = 16;  // 4 * sizeof(float)
    msg.row_step = msg.point_step * msg.width;
    msg.data.resize(msg.row_step);
    
    // Fill point cloud data
    auto points_accessor = points_cpu.accessor<float, 2>();
    auto sem_accessor = sem_cpu.accessor<int, 1>();
    auto ins_accessor = ins_cpu.accessor<int, 1>();
    
    for (int i = 0; i < num_points; ++i) {
        uint8_t* point_data = &msg.data[i * msg.point_step];
        
        // Position
        *reinterpret_cast<float*>(point_data + 0) = points_accessor[i][0];
        *reinterpret_cast<float*>(point_data + 4) = points_accessor[i][1];
        *reinterpret_cast<float*>(point_data + 8) = points_accessor[i][2];
        
        // Color based on semantic class
        int class_id = sem_accessor[i];
        auto color = getColorForClass(class_id);
        
        // Pack RGB into float (PCL format)
        uint32_t rgb = ((uint32_t)color[0] << 16 | (uint32_t)color[1] << 8 | (uint32_t)color[2]);
        *reinterpret_cast<float*>(point_data + 12) = *reinterpret_cast<float*>(&rgb);
    }
    
    return msg;
}

std::array<uint8_t, 3> LidarProcessor::getColorForClass(int class_id)
{
    if (semantic_map_.find(class_id) != semantic_map_.end()) {
        auto &color = semantic_map_[class_id].color;
        return {
            static_cast<uint8_t>(color[0] * 255),
            static_cast<uint8_t>(color[1] * 255),
            static_cast<uint8_t>(color[2] * 255)
        };
    }
    // Default color (gray)
    return {128, 128, 128};
}

std::array<uint8_t, 3> LidarProcessor::getInstanceColor(int instance_id)
{
    // Generate distinct colors for instances using hash
    if (instance_id == 0) return {0, 0, 0};  // Background
    
    std::hash<int> hasher;
    size_t hash = hasher(instance_id);
    
    uint8_t r = (hash & 0xFF0000) >> 16;
    uint8_t g = (hash & 0x00FF00) >> 8;
    uint8_t b = (hash & 0x0000FF);
    
    // Ensure minimum brightness
    if (r < 50 && g < 50 && b < 50) {
        r += 100; g += 100; b += 100;
    }
    
    return {r, g, b};
}

}  // namespace lidar_processing

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(lidar_processing::LidarProcessor)
