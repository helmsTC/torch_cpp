#include "lidar/lidar.h"
#include <memory>
#include <chrono>
#include <random>

namespace lidar_processing
{

LidarProcessor::LidarProcessor(const rclcpp::NodeOptions &options)
    : Node("lidar_processor", options),
      model_loaded_(false),
      num_classes_(20),  // KITTI default
      overlap_threshold_(0.8),
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
    
    model_path_ = this->get_parameter("model_path").as_string();
    std::string input_topic = this->get_parameter("input_topic").as_string();
    use_cuda_ = this->get_parameter("use_cuda").as_bool();
    batch_size_ = this->get_parameter("batch_size").as_int();
    num_classes_ = this->get_parameter("num_classes").as_int();
    overlap_threshold_ = this->get_parameter("overlap_threshold").as_double();
    
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
        int num_points = 1000;
        torch::Tensor dummy_points = torch::randn({num_points, 3}, device_);
        torch::Tensor dummy_features = torch::randn({num_points, 4}, device_);
        
        // The MaskPLS model expects a dictionary with 'pt_coord' and 'feats' as lists
        // In C++, we need to use GenericDict or pass inputs differently
        
        // Method 1: Try passing as a tuple (if model supports it)
        std::vector<torch::jit::IValue> inputs;
        
        // Create lists of tensors
        c10::List<torch::Tensor> pt_coord_list;
        pt_coord_list.push_back(dummy_points);
        
        c10::List<torch::Tensor> feats_list;
        feats_list.push_back(dummy_features);
        
        // Create a generic dictionary
        c10::impl::GenericDict dict(c10::StringType::get(), c10::AnyType::get());
        dict.insert("pt_coord", pt_coord_list);
        dict.insert("feats", feats_list);
        
        inputs.push_back(dict);
        
        RCLCPP_INFO(this->get_logger(), "Testing model with %d points", num_points);
        
        // Forward pass
        auto output = model_->forward(inputs);
        
        RCLCPP_INFO(this->get_logger(), "Model test successful!");
        
    } catch (const std::exception& e) {
        RCLCPP_WARN(this->get_logger(), "Model test warning: %s", e.what());
        RCLCPP_INFO(this->get_logger(), "Model loaded but test forward pass failed. This might be normal if the model expects specific input format.");
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
    auto tensor_pair = pointcloud2ToTensor(msg);
    torch::Tensor points_tensor = tensor_pair.first;
    torch::Tensor features_tensor = tensor_pair.second;
    
    if (!points_tensor.defined() || points_tensor.numel() == 0) {
        RCLCPP_WARN(this->get_logger(), "Failed to convert pointcloud to tensor");
        return;
    }
    
    // Process with model
    try {
        auto result_pair = processWithModel(points_tensor, features_tensor);
        torch::Tensor semantic_pred = result_pair.first;
        torch::Tensor instance_pred = result_pair.second;
        
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
        return std::make_pair(torch::Tensor(), torch::Tensor());
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
    
    // Move to device
    points = points.to(device_);
    features = features.to(device_);
    
    return std::make_pair(points, features);
}

std::pair<torch::Tensor, torch::Tensor> LidarProcessor::processWithModel(
    const torch::Tensor &points, const torch::Tensor &features)
{
    torch::NoGradGuard no_grad;
    
    try {
        // Create input for the model
        std::vector<torch::jit::IValue> inputs;
        
        // Create lists of tensors (MaskPLS expects batch format)
        c10::List<torch::Tensor> pt_coord_list;
        pt_coord_list.push_back(points.cpu());  // Model might expect CPU tensors
        
        c10::List<torch::Tensor> feats_list;
        feats_list.push_back(features.cpu());
        
        // Create a generic dictionary
        c10::impl::GenericDict dict(c10::StringType::get(), c10::AnyType::get());
        dict.insert("pt_coord", pt_coord_list);
        dict.insert("feats", feats_list);
        
        inputs.push_back(dict);
        
        // Forward pass
        auto output = model_->forward(inputs);
        
        // Parse output - expecting tuple of (outputs_dict, padding, bb_logits)
        if (!output.isTuple()) {
            RCLCPP_ERROR(this->get_logger(), "Expected tuple output from model");
            // Return dummy outputs
            torch::Tensor zero_sem = torch::zeros({points.size(0)}, torch::kInt32);
            torch::Tensor zero_ins = torch::zeros({points.size(0)}, torch::kInt32);
            return std::make_pair(zero_sem, zero_ins);
        }
        
        auto output_tuple = output.toTuple();
        if (output_tuple->elements().size() < 2) {
            RCLCPP_ERROR(this->get_logger(), "Model output tuple has insufficient elements");
            torch::Tensor zero_sem = torch::zeros({points.size(0)}, torch::kInt32);
            torch::Tensor zero_ins = torch::zeros({points.size(0)}, torch::kInt32);
            return std::make_pair(zero_sem, zero_ins);
        }
        
        // Get outputs dictionary and padding
        auto outputs_dict = output_tuple->elements()[0].toGenericDict();
        auto padding = output_tuple->elements()[1].toTensor();
        
        // Extract predictions
        auto pred_logits = outputs_dict.at("pred_logits").toTensor();
        auto pred_masks = outputs_dict.at("pred_masks").toTensor();
        
        // Perform simple semantic inference for now
        // (Full panoptic inference would require more complex processing)
        torch::Tensor semantic_pred = simplifiedSemanticInference(pred_logits, pred_masks, padding);
        torch::Tensor instance_pred = torch::zeros_like(semantic_pred);
        
        return std::make_pair(semantic_pred, instance_pred);
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in model processing: %s", e.what());
        // Return zeros on error
        torch::Tensor zero_sem = torch::zeros({points.size(0)}, torch::kInt32);
        torch::Tensor zero_ins = torch::zeros({points.size(0)}, torch::kInt32);
        return std::make_pair(zero_sem, zero_ins);
    }
}

torch::Tensor LidarProcessor::simplifiedSemanticInference(
    const torch::Tensor &pred_logits,
    const torch::Tensor &pred_masks,
    const torch::Tensor &padding)
{
    // Simplified semantic inference
    int batch_size = pred_logits.size(0);
    
    if (batch_size == 0) {
        return torch::zeros(0, torch::kInt32);
    }
    
    // Process first batch element
    auto mask_cls = pred_logits[0];  // [num_queries, num_classes+1]
    auto mask_pred = pred_masks[0];  // [num_points, num_queries]
    auto pad = padding[0];  // [num_points]
    
    // Get valid points (not padded)
    auto valid_mask = ~pad;
    auto valid_points = valid_mask.sum().item<int>();
    
    if (valid_points == 0) {
        return torch::zeros(mask_pred.size(0), torch::kInt32);
    }
    
    // Get predictions from masks
    mask_pred = mask_pred.index({valid_mask}).sigmoid();
    
    // Simple argmax semantic segmentation
    auto mask_cls_prob = mask_cls.softmax(-1);
    mask_cls_prob = mask_cls_prob.index({"...", torch::indexing::Slice(0, num_classes_)});
    
    // Compute semantic predictions
    auto semantic_seg = torch::zeros({valid_points}, torch::kInt32).to(device_);
    
    if (mask_pred.size(1) > 0) {
        // Weight masks by class probabilities
        auto weighted_masks = torch::einsum("qc,pq->pc", {mask_cls_prob, mask_pred});
        semantic_seg = weighted_masks.argmax(1).to(torch::kInt32);
    }
    
    // Create full output with padding
    auto full_semantic = torch::zeros({mask_pred.size(0)}, torch::kInt32).to(device_);
    full_semantic.index_put_({valid_mask}, semantic_seg);
    
    return full_semantic.cpu();
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

}  // namespace lidar_processing

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(lidar_processing::LidarProcessor)
